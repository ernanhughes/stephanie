# stephanie/agents/learning_from_learning.py
from __future__ import annotations

import time
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent

# Reuse your existing pieces (present in your repo bundle)
from stephanie.agents.knowledge.chat_annotate import ChatAnnotateAgent
from stephanie.agents.knowledge.chat_analyze import ChatAnalyzeAgent
from stephanie.agents.knowledge.conversation_filter import ConversationFilterAgent
from stephanie.agents.knowledge.chat_knowledge_builder import ChatKnowledgeBuilder


# Optional: if present in your tree; the agent will fall back if not
from stephanie.scoring.scorer.knowledge_scorer import KnowledgeScorer  # two-head scorer

_logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    verification_threshold: float = 0.85
    skeptic_weight: float = 0.34
    editor_weight: float = 0.33
    risk_weight: float = 0.33
    version: int = 1


class LearningFromLearningAgent(BaseAgent):
    """
    Minimal, production-ready “Learning from Learning” agent.

    Pipeline:
      1) Persist & annotate: domain + NER, plus AI judge pass (triage).
      2) Filter: keep critical messages for the current paper/section.
      3) Build knowledge: chat/paper units (domains, entities, anchors, KG links).
      4) Summarize → verify → improve (iterative, strategy-weighted).
      5) Persist pairs for DPO-lite; (optional) train on schedule.

    Inputs in `context`:
      paper_data: { id|doc_id, title, abstract, sections?: [{section_name, section_text, domain?}] }
      chat_corpus: [ {id?, role, text, timestamp?}, ... ]
      section_name?: str   # if you want to focus on a single section
      max_iterations?: int # default 3
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Sub-agents / utilities
        self.annotate = ChatAnnotateAgent(cfg.get("annotate", {}), memory, container, logger)
        self.analyze  = ChatAnalyzeAgent(cfg.get("analyze",  {}), memory, container, logger)
        self.filter   = ConversationFilterAgent(cfg.get("filter",   {}), memory, container, logger)
        self.builder  = ChatKnowledgeBuilder(cfg.get("builder",  {}), memory, container, logger)

        self.strategy = Strategy(
            verification_threshold = cfg.get("verification_threshold", 0.85),
            skeptic_weight         = cfg.get("skeptic_weight", 0.34),
            editor_weight          = cfg.get("editor_weight", 0.33),
            risk_weight            = cfg.get("risk_weight", 0.33),
            version                = 1
        )

        # Optional two-head scorer
        self.knowledge_scorer = None
        if KnowledgeScorer is not None:
            try:
                self.knowledge_scorer = KnowledgeScorer(cfg.get("knowledge_scorer", {}), memory, container, logger)
            except Exception as e:
                _logger.warning(f"KnowledgeScorer unavailable, falling back: {e}")

    # ---------------- main ----------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()

        
        paper = context.get("paper_data", {}) or {}
        chat  = context.get("chat_corpus", []) or []
        section_name = context.get("section_name")

        # 0) fast-exit guard
        if not paper.get("abstract"):
            raise ValueError("paper_data.abstract is required")
        if not chat:
            _logger.warning("No chat_corpus provided; proceeding with paper-only path")

        # 1) Persist & annotate chats (idempotent)
        await self.annotate.run({})
        await self.analyze.run({"limit": self.cfg.get("judge_limit", 8000)})

        # 2) Section focus + filter (critical path)
        section = self._select_section(paper, section_name)
        fctx = await self.filter.run({
            "paper_section": section,
            "chat_corpus": chat,
            "goal_template": "academic_summary"
        })
        critical_msgs = fctx.get("critical_messages", [])
        context.update({
            "filter_threshold": fctx.get("filter_threshold"),
            "critical_messages": critical_msgs,
            "critical_path": fctx.get("critical_path", []),
            "section_domain": fctx.get("section_domain"),
        })

        # 3) Build knowledge units (chat + paper); adds domains/entities/anchors/KG links
        ku = self.builder.build(
            chat_messages=critical_msgs if critical_msgs else chat,
            paper_text=section["section_text"],
            conversation_id=context.get("conversation_id"),
            context=context
        )
        context["knowledge_units"] = {k: v.to_dict() for k, v in ku.items()}

        # 4) Baseline summary → verify → improve (few steps)
        baseline = await self._baseline_summary(paper, section, critical_msgs, context)
        verify   = await self._verify_and_improve(baseline, paper, section, context)

        # 5) Persist DPO-lite pairs for later training
        self._persist_pairs(paper_id=paper.get("id") or paper.get("doc_id"),
                            baseline=baseline,
                            improved=verify["summary"],
                            metrics=verify["metrics"])

        result = {
            "summary": verify["summary"],
            "metrics": verify["metrics"],
            "iterations": verify["iterations"],
            "strategy": vars(self.strategy),
            "duration_sec": round(time.time() - t0, 2),
        }
        self.logger.log("LfL_Run_Complete", result)
        return {**context, **result}

    # ---------------- steps ----------------

    def _select_section(self, paper: Dict[str, Any], prefer: Optional[str]) -> Dict[str, str]:
        sections = paper.get("sections") or []
        if prefer:
            for s in sections:
                if s.get("section_name", "").lower() == prefer.lower():
                    return s
        # fallback: synthesize from title+abstract
        return {
            "section_name": prefer or "Abstract",
            "section_text": f"{paper.get('title','').strip()}\n\n{paper.get('abstract','').strip()}",
            "domain": "general"
        }

    async def _baseline_summary(self, paper: Dict[str, Any], section: Dict[str, Any],
                                critical_msgs: List[Dict[str, Any]], ctx: Dict[str, Any]) -> str:
        # very small prompt; swap with your SimplePaperSummarizer if desired
        prompt = self.prompt_loader.load_prompt("baseline_summary", {
            "title": paper.get("title",""),
            "abstract": paper.get("abstract",""),
            "focus_section": section.get("section_name"),
            "focus_text": section.get("section_text","")[:5000],
            "hints": "\n".join(m["text"] for m in critical_msgs[:6]) if critical_msgs else "",
        })
        return await self.call_llm(prompt, ctx)

    async def _verify_and_improve(self, summary: str, paper: Dict[str, Any],
                                  section: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        max_iter = int(self.cfg.get("max_iterations", 3))
        current = summary
        iters: List[Dict[str, Any]] = []

        for i in range(1, max_iter + 1):
            metrics = self._score_summary(current, paper, section, ctx)

            iters.append({"iteration": i, "score": metrics["overall"], "metrics": metrics})
            if metrics["overall"] >= self.strategy.verification_threshold:
                break

            # light, strategy-weighted improvement step
            improve_prompt = self.prompt_loader.load_prompt("improve_summary", {
                "title": paper.get("title",""),
                "section_name": section.get("section_name"),
                "section_text": section.get("section_text","")[:6000],
                "current_summary": current,
                "skeptic_weight": self.strategy.skeptic_weight,
                "editor_weight":  self.strategy.editor_weight,
                "risk_weight":    self.strategy.risk_weight,
                "weaknesses": json.dumps(metrics.get("weaknesses", []), ensure_ascii=False),
            })
            current = await self.call_llm(improve_prompt, ctx)

        # evolve simple strategy between papers (meta-adapt)
        self._evolve_strategy(iters)

        return {"summary": current, "metrics": metrics, "iterations": iters}

    # ---------------- scoring (two-head if available, else heuristic) ----------------

    def _score_summary(self, text: str, paper: Dict[str, Any],
                       section: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        if self.knowledge_scorer:
            # goal = title/abstract; candidate = summary
            goal_text = f"{paper.get('title','')}\n\n{paper.get('abstract','')}"
            p, details = self.knowledge_scorer.predict(
                goal_text=goal_text, candidate_text=text,
                meta={"text_len_norm": min(1.0, len(text)/2000.0)},
                return_components=True
            )
            knowledge = float(details.get("probability", p))
            # quick rubric for secondary dims
            clarity, grounding = self._rubric_dims(text, section.get("section_text",""))
            overall = 0.6*knowledge + 0.25*clarity + 0.15*grounding
            weaknesses = self._weaknesses(text, section.get("section_text",""))
            return {"overall": overall, "knowledge_score": knowledge,
                    "clarity": clarity, "grounding": grounding, "weaknesses": weaknesses}
        else:
            # simple heuristic fallback
            clarity, grounding = self._rubric_dims(text, section.get("section_text",""))
            knowledge = 0.5*clarity + 0.5*grounding
            overall = 0.6*knowledge + 0.25*clarity + 0.15*grounding
            weaknesses = self._weaknesses(text, section.get("section_text",""))
            return {"overall": overall, "knowledge_score": knowledge,
                    "clarity": clarity, "grounding": grounding, "weaknesses": weaknesses}

    def _rubric_dims(self, text: str, ref: str) -> Tuple[float, float]:
        # clarity: sentence length, concrete terms
        import re
        sents = [s for s in re.split(r'[.!?]\s+', text.strip()) if s]
        avg_len = sum(len(s.split()) for s in sents)/max(1,len(sents))
        clarity = max(0.0, min(1.0, 1.1 - (abs(avg_len-22)/22)))  # bell around ~22 words
        # grounding: lexical overlap proxy
        def toks(t): return set(re.findall(r'\b\w+\b', t.lower()))
        inter = len(toks(text) & toks(ref))
        grounding = max(0.0, min(1.0, inter / max(30, len(toks(ref)) or 1)))
        return clarity, grounding

    def _weaknesses(self, summary: str, ref: str) -> List[str]:
        out = []
        if len(summary) < 400: out.append("too short / thin detail")
        if "we propose" in ref.lower() and "we propose" not in summary.lower():
            out.append("misses core claim language")
        if summary.count("(") != summary.count(")"):
            out.append("formatting/parens issues")
        return out

    # ---------------- meta-learning (very simple & transparent) ----------------

    def _evolve_strategy(self, iters: List[Dict[str, Any]]):
        if len(iters) < 2:  # nothing to learn
            return
        gains = [iters[i]["score"] - iters[i-1]["score"] for i in range(1, len(iters))]
        avg_gain = sum(gains)/len(gains) if gains else 0.0

        changed = False
        if avg_gain < self.cfg.get("min_gain", 0.01):  # stagnating → add skepticism
            self.strategy.skeptic_weight = min(0.60, self.strategy.skeptic_weight + 0.06)
            self.strategy.editor_weight  = max(0.20, self.strategy.editor_weight  - 0.03)
            self.strategy.risk_weight    = max(0.20, self.strategy.risk_weight    - 0.03)
            changed = True
        elif avg_gain > self.cfg.get("high_gain", 0.03):  # cruising → lower threshold slightly
            self.strategy.verification_threshold = max(0.80, self.strategy.verification_threshold - 0.01)
            changed = True

        if changed:
            self.strategy.version += 1
            self.logger.log("LfL_Strategy_Evolved", {
                "avg_gain": round(avg_gain, 4),
                "strategy": vars(self.strategy)
            })

    # ---------------- DPO-lite pair persistence ----------------

    def _persist_pairs(self, paper_id: Any, baseline: str, improved: str, metrics: Dict[str, Any]):
        """
        Save (A=improved, B=baseline) as scorable pairs so KnowledgePairBuilder
        can sweep them later. Uses your existing memory interfaces.
        """
        try:
            if not paper_id:  # best-effort
                paper_id = f"paper:{hash((baseline, improved))}"
            # Positive
            self.memory.casebooks.add_scorable(
                case_id=str(paper_id),
                role="knowledge_pair_positive",
                text=improved,
                meta={"verification_score": metrics.get("overall", 0.0),
                      "knowledge_score": metrics.get("knowledge_score", 0.0),
                      "strategy_version": self.strategy.version}
            )
            # Negative (if meaningfully worse)
            if metrics.get("overall", 0.0) >= self.strategy.verification_threshold:
                self.memory.casebooks.add_scorable(
                    case_id=str(paper_id),
                    role="knowledge_pair_negative",
                    text=baseline,
                    meta={"verification_score": max(0.0, metrics.get("overall", 0.0) - 0.15),
                          "knowledge_score": max(0.0, metrics.get("knowledge_score", 0.0) * 0.7),
                          "strategy_version": self.strategy.version}
                )
        except Exception as e:
            _logger.warning(f"Pair persistence skipped: {e}")

