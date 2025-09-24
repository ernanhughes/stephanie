# stephanie/agents/learning_from_learning.py
from __future__ import annotations

import time
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.knowledge.scorable_annotate import ScorableAnnotateAgent
from stephanie.models.casebook import CaseBookORM  # NEW
from stephanie.utils.casebook_utils import generate_casebook_name  # NEW
from stephanie.utils.paper_utils import (                      # NEW
    build_paper_goal_meta,
    build_paper_goal_text,
    section_goal_text,
    section_quality,
    system_guidance_from_goal,
)

# Reuse your existing pieces (present in your repo bundle)
from stephanie.agents.knowledge.chat_annotate import ChatAnnotateAgent
from stephanie.agents.knowledge.chat_analyze import ChatAnalyzeAgent
from stephanie.agents.knowledge.conversation_filter import ConversationFilterAgent
from stephanie.agents.knowledge.chat_knowledge_builder import ChatKnowledgeBuilder

# Optional two-head scorer
from stephanie.scoring.scorer.knowledge_scorer import KnowledgeScorer

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
    Production-ready “Learning from Learning” agent with DSPy-style paper/section
    preprocessing (structured sections, goals, casebooks) before the verification loop.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Sub-agents / utilities
        self.annotate = ScorableAnnotateAgent(cfg.get("annotate", {}), memory, container, logger)
        self.analyze  = ChatAnalyzeAgent(cfg.get("analyze",  {}), memory, container, logger)
        self.filter   = ConversationFilterAgent(cfg.get("filter",   {}), memory, container, logger)
        self.builder  = ChatKnowledgeBuilder(cfg.get("builder",  {}), memory, container, logger)

        # Config lifted from DSPyPaperSectionProcessorAgent (names preserved)
        self.max_refinements      = int(cfg.get("max_iterations", 3))
        self.min_section_length   = int(cfg.get("min_section_length", 100)) 
        self.casebook_action      = cfg.get("casebook_action", "blog")      
        self.goal_template        = cfg.get("goal_template", "academic_summary") 

        self.strategy = Strategy(
            verification_threshold = cfg.get("verification_threshold", 0.85),
            skeptic_weight         = cfg.get("skeptic_weight", 0.34),
            editor_weight          = cfg.get("editor_weight", 0.33),
            risk_weight            = cfg.get("risk_weight", 0.33),
            version                = 1
        )

        self.corpus = self.container.get_service("chat_corpus")  # for chat triage

        # Optional two-head scorer
        self.knowledge_scorer = None
        try:
            self.knowledge_scorer = KnowledgeScorer(cfg.get("knowledge_scorer", {}), memory, container, logger)
        except Exception as e:
            _logger.warning(f"KnowledgeScorer unavailable, falling back: {e}")

    # ---------------- public entry ----------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Paper-level runner (multi-doc aware). Mirrors the DSPy paper agent:
          - CaseBook per paper
          - Goal per paper
          - Structured sections (or synth from title+abstract)
          - Section loop with filtering → knowledge build → baseline → verify+improve
          - Persist each section’s artifacts & metrics
        """
        t_run = self._t0()
        documents = context.get(self.input_key, []) 
        processed_sections = []
        self.report({"event": "input", "agent": self.name, "docs_count": len(documents)})
        # Inputs
        chat  = context.get("chat_corpus", []) or []

        # Process each document
        for di, paper in enumerate(documents, start=1):

            section_focus = context.get("section_name")
            doc_id = paper.get("scorable_id") 
            structured_sections = self.memory.document_sections.get_by_document(doc_id) or []

            abstract = None
            for sec in structured_sections:
                if sec.section_name.lower() == "abstract":
                    abstract = sec.section_text
                    break


            if not abstract:
                raise ValueError("paper_data.abstract is required")

            # 1) Persist & annotate chats (idempotent triage)
            await self.annotate.run(context={"scorables":[paper]})
            await self.analyze.run({"limit": self.cfg.get("judge_limit", 8000)})

            # 2) Create CaseBook + Goal for this paper
            title   = paper.get("title", "")
            doc_id  = paper.get("id") or paper.get("doc_id") or f"paper:{hash(title)}"

            casebook_name = generate_casebook_name(self.casebook_action, title)
            casebook = self.memory.casebooks.ensure_casebook(
                name=casebook_name,
                description=f"LfL agent runs for paper {title}",
                tag=self.casebook_action,
            )

            paper_goal = self.memory.goals.get_or_create(
                {
                    "goal_text": build_paper_goal_text(title),
                    "description": "Learning-from-learning: verify & improve per section.",
                    "meta": build_paper_goal_meta(title, doc_id, domains=self.cfg.get("domains", [])),
                }
            ).to_dict()

            # 3) Resolve sections (structured first; fallback to Abstract synth)
            structured_sections = self.memory.document_sections.get_by_document(doc_id) or []
            if structured_sections:
                sections_todo: List[Dict[str, Any]] = []
                for sec in structured_sections:
                    if section_focus and sec.section_name.lower() != section_focus.lower():
                        continue
                    txt = sec.section_text or ""
                    if len(txt) < self.min_section_length:
                        continue
                    sections_todo.append({
                        "section_name": sec.section_name,
                        "section_text": txt,
                        "section_id": sec.id,
                        "order_index": getattr(sec, "order_index", None),
                    })
                if not sections_todo:
                    # fallback to Abstract synth if filters removed everything
                    sections_todo = [self._synth_abstract_section(paper, prefer=section_focus)]
            else:
                sections_todo = [self._synth_abstract_section(paper, prefer=section_focus)]

            results: List[Dict[str, Any]] = []

            # 4) Per-section loop
            for si, section in enumerate(sections_todo, start=1):
                st0 = self._t0()
                section_name = section["section_name"]
                section_text = section["section_text"]

                # Filter conversations for this section’s topic
                fctx = await self.filter.run({
                    "paper_section": section,
                    "chat_corpus": chat,
                    "goal_template": self.goal_template
                })
                critical_msgs = fctx.get("critical_messages", []) or chat
                section_domain = fctx.get("section_domain")

                # Build knowledge units (chat + paper)
                ku = self.builder.build(
                    chat_messages=critical_msgs,
                    paper_text=section_text,
                    conversation_id=context.get("conversation_id"),
                    context={**context, "section_name": section_name}
                )

                # Baseline summary
                baseline = await self._baseline_summary(paper, section, critical_msgs, context)

                # Verify & improve (few iterations)
                verify = await self._verify_and_improve(baseline, paper, section, context)

                # Persist artifacts to casebook (DSPy style)
                self._save_section_to_casebook(
                    casebook=casebook,
                    goal_id=paper_goal["id"],
                    doc_id=str(doc_id),
                    section_name=section_name, 
                    section_text=section_text,
                    result={
                        "initial_draft": {"title": section_name, "body": baseline},
                        "refined_draft": {"title": section_name, "body": verify["summary"]},
                        "verification_report": {"scores": verify["metrics"], "iterations": verify["iterations"]},
                        "final_validation": {"scores": verify["metrics"], "passed": verify["metrics"]["overall"] >= self.strategy.verification_threshold},
                        "passed": verify["metrics"]["overall"] >= self.strategy.verification_threshold,
                        "refinement_iterations": len(verify["iterations"]),
                    },
                    context={
                        **context,
                        "paper_title": title,
                        "paper_id": doc_id,
                        "section_order_index": section.get("order_index"),
                    },
                )

                # Persist DPO-lite pairs for later training
                self._persist_pairs(
                    paper_id=doc_id,
                    baseline=baseline,
                    improved=verify["summary"],
                    metrics=verify["metrics"],
                )

                results.append({
                    "section_name": section_name,
                    "summary": verify["summary"],
                    "metrics": verify["metrics"],
                    "iterations": verify["iterations"],
                    "elapsed_ms": self._ms_since(st0),
                    "domain": section_domain,
                })

            # Done
            out = {
                "paper_id": doc_id,
                "title": title,
                "results": results,
                "strategy": vars(self.strategy),
                "elapsed_ms": self._ms_since(t_run),
            }
            self.logger.log("LfL_Paper_Run_Complete", out)
        return {**context, **out}

    # ---------------- section/prep borrowed from DSPy agent ----------------

    def _synth_abstract_section(self, paper: Dict[str, Any], prefer: Optional[str]) -> Dict[str, Any]:
        """Fallback section built from title+abstract, mirroring DSPy agent behavior."""
        return {
            "section_name": prefer or "Abstract",
            "section_text": f"{paper.get('title','').strip()}\n\n{paper.get('abstract','').strip()}",
            "order_index": None,
        }

    async def _baseline_summary(self, paper: Dict[str, Any], section: Dict[str, Any],
                                critical_msgs: List[Dict[str, Any]], ctx: Dict[str, Any]) -> str:
        prompt = self.prompt_loader.load_prompt("baseline_summary", {
            "title": paper.get("title",""),
            "abstract": paper.get("abstract",""),
            "focus_section": section.get("section_name"),
            "focus_text": section.get("section_text","")[:5000],
            "hints": "\n".join(m.get("text","") for m in (critical_msgs[:6] if critical_msgs else [])),
        })
        return await self.call_llm(prompt, ctx)

    async def _verify_and_improve(self, summary: str, paper: Dict[str, Any],
                                  section: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        max_iter = self.max_refinements
        current = summary
        iters: List[Dict[str, Any]] = []

        for i in range(1, max_iter + 1):
            metrics = self._score_summary(current, paper, section, ctx)
            iters.append({"iteration": i, "score": metrics["overall"], "metrics": metrics})

            if metrics["overall"] >= self.strategy.verification_threshold:
                break

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

        # Meta-adapt *between* sections/papers
        self._evolve_strategy(iters)
        return {"summary": current, "metrics": metrics, "iterations": iters}

    # ---------------- scoring (two-head if available) ----------------

    def _score_summary(self, text: str, paper: Dict[str, Any],
                       section: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        if self.knowledge_scorer:
            goal_text = f"{paper.get('title','')}\n\n{paper.get('abstract','')}"
            p, comps = self.knowledge_scorer.model.predict(
                goal_text,
                text,
                meta={"text_len_norm": min(1.0, len(text)/2000.0)},
                return_components=True,
            )
            knowledge = float(comps.get("probability", p))
            clarity, grounding = self._rubric_dims(text, section.get("section_text",""))
            overall = 0.6*knowledge + 0.25*clarity + 0.15*grounding
            weaknesses = self._weaknesses(text, section.get("section_text",""))
            return {"overall": overall, "knowledge_score": knowledge,
                    "clarity": clarity, "grounding": grounding, "weaknesses": weaknesses}
        else:
            clarity, grounding = self._rubric_dims(text, section.get("section_text",""))
            knowledge = 0.5*clarity + 0.5*grounding
            overall = 0.6*knowledge + 0.25*clarity + 0.15*grounding
            weaknesses = self._weaknesses(text, section.get("section_text",""))
            return {"overall": overall, "knowledge_score": knowledge,
                    "clarity": clarity, "grounding": grounding, "weaknesses": weaknesses}

    def _rubric_dims(self, text: str, ref: str) -> Tuple[float, float]:
        import re
        sents = [s for s in re.split(r'[.!?]\s+', text.strip()) if s]
        avg_len = sum(len(s.split()) for s in sents) / max(1, len(sents))
        clarity = max(0.0, min(1.0, 1.1 - (abs(avg_len - 22) / 22)))
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

    # ---------------- meta-learning ----------------

    def _evolve_strategy(self, iters: List[Dict[str, Any]]):
        if len(iters) < 2:
            return
        gains = [iters[i]["score"] - iters[i-1]["score"] for i in range(1, len(iters))]
        avg_gain = sum(gains)/len(gains) if gains else 0.0

        changed = False
        if avg_gain < self.cfg.get("min_gain", 0.01):
            self.strategy.skeptic_weight = min(0.60, self.strategy.skeptic_weight + 0.06)
            self.strategy.editor_weight  = max(0.20, self.strategy.editor_weight  - 0.03)
            self.strategy.risk_weight    = max(0.20, self.strategy.risk_weight    - 0.03)
            changed = True
        elif avg_gain > self.cfg.get("high_gain", 0.03):
            self.strategy.verification_threshold = max(0.80, self.strategy.verification_threshold - 0.01)
            changed = True

        if changed:
            self.strategy.version += 1
            self.logger.log("LfL_Strategy_Evolved", {
                "avg_gain": round(avg_gain, 4),
                "strategy": vars(self.strategy)
            })

    # ---------------- persistence (DSPy-style) ----------------

    def _save_section_to_casebook(
        self,
        casebook: CaseBookORM,
        goal_id: int,
        doc_id: str,
        section_name: str,
        section_text: str,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ):
        """Mirror DSPy persistence: prompt meta + per-artifact scorables."""
        paper_title = context.get("paper_title")
        order_index = context.get("section_order_index")
        pipeline_run_id = context.get("pipeline_run_id")

        case_prompt = {
            "paper_id": doc_id,
            "paper_title": paper_title,
            "section_name": section_name,
            "section_order_index": order_index,
        }
        case_meta = {
            "type": "draft_trajectory",
            "paper_id": doc_id,
            "paper_title": paper_title,
            "section_name": section_name,
            "section_order_index": order_index,
            "timestamp": time.time(),
            "source": "lfl.agent",
        }
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=goal_id,
            prompt_text=self._dumps_safe(case_prompt),
            agent_name=self.name,
            meta=case_meta,
        )

        def _smeta(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
            base = {
                "paper_id": doc_id,
                "paper_title": paper_title,
                "section_name": section_name,
                "section_order_index": order_index,
            }
            if extra: base.update(extra)
            return base

        self.memory.casebooks.add_scorable(
            case_id=case.id, pipeline_run_id=pipeline_run_id,
            text=section_text, role="section_text", meta=_smeta()
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id, pipeline_run_id=pipeline_run_id,
            text=self._dumps_safe(result.get("initial_draft", {})),
            role="initial_draft", meta=_smeta()
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id, pipeline_run_id=pipeline_run_id,
            text=self._dumps_safe(result.get("refined_draft", {})),
            role="refined_draft",
            meta=_smeta({"refinement_iterations": result.get("refinement_iterations", 0)})
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id, pipeline_run_id=pipeline_run_id,
            text=self._dumps_safe(result.get("verification_report", {})),
            role="verification_report", meta=_smeta()
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id, pipeline_run_id=pipeline_run_id,
            text=self._dumps_safe(result.get("final_validation", {})),
            role="final_validation", meta=_smeta()
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id, pipeline_run_id=pipeline_run_id,
            text=self._dumps_safe({
                "passed": result.get("passed", False),
                "refinement_iterations": result.get("refinement_iterations", 0),
                "final_scores": (result.get("final_validation", {}) or {}).get("scores", {}),
            }),
            role="metrics", meta=_smeta()
        )

    # ---------------- DPO-lite pair persistence ----------------

    def _persist_pairs(self, paper_id: Any, baseline: str, improved: str, metrics: Dict[str, Any]):
        try:
            if not paper_id:
                paper_id = f"paper:{hash((baseline, improved))}"
            self.memory.casebooks.add_scorable(
                case_id=str(paper_id),
                role="knowledge_pair_positive",
                text=improved,
                meta={"verification_score": metrics.get("overall", 0.0),
                      "knowledge_score": metrics.get("knowledge_score", 0.0),
                      "strategy_version": self.strategy.version}
            )
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

    # ---------------- small helpers (ported) ----------------

    def _t0(self): 
        return time.time()

    def _ms_since(self, t0):
        return round((time.time() - t0) * 1000, 1)

    def _dumps_safe(self, obj) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json.dumps({"_warning": "failed_to_dump", "repr": repr(obj)})
