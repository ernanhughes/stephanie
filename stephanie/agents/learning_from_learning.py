# stephanie/agents/learning_from_learning.py
from __future__ import annotations

import time
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.knowledge.scorable_annotate import ScorableAnnotateAgent
from stephanie.models.casebook import CaseBookORM, CaseORM
from stephanie.models.document_section import DocumentSectionORM
from stephanie.scoring.scorable import ScorableType
from stephanie.tools.chat_corpus_tool import build_chat_corpus_tool
from stephanie.utils.casebook_utils import generate_casebook_name
from stephanie.utils.paper_utils import (
    build_paper_goal_meta,
    build_paper_goal_text,
)

# Reuse your existing pieces (present in your repo bundle)
from stephanie.agents.knowledge.chat_analyze import ChatAnalyzeAgent
from stephanie.agents.knowledge.conversation_filter import (
    ConversationFilterAgent,
)
from stephanie.agents.knowledge.chat_knowledge_builder import (
    ChatKnowledgeBuilder,
)

# REMOVE this line:
# self.corpus = self.container.get_service("chat_corpus")  # for chat triage


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
        self.annotate = ScorableAnnotateAgent(
            cfg.get("annotate", {}), memory, container, logger
        )
        self.analyze = ChatAnalyzeAgent(
            cfg.get("analyze", {}), memory, container, logger
        )
        self.filter = ConversationFilterAgent(
            cfg.get("filter", {}), memory, container, logger
        )
        self.builder = ChatKnowledgeBuilder(
            cfg.get("builder", {}), memory, container, logger
        )

        # Config lifted from DSPyPaperSectionProcessorAgent (names preserved)
        self.max_refinements = int(cfg.get("max_iterations", 3))
        self.min_section_length = int(cfg.get("min_section_length", 100))
        self.casebook_action = cfg.get("casebook_action", "blog")
        self.goal_template = cfg.get("goal_template", "academic_summary")

        self.strategy = Strategy(
            verification_threshold=cfg.get("verification_threshold", 0.85),
            skeptic_weight=cfg.get("skeptic_weight", 0.34),
            editor_weight=cfg.get("editor_weight", 0.33),
            risk_weight=cfg.get("risk_weight", 0.33),
            version=1,
        )

        # ADD this:
        self.chat_corpus = build_chat_corpus_tool(
            memory=memory,
            container=container,
            cfg=cfg.get("chat_corpus", {}),  # optional weights, k, etc.
        )

        # Optional two-head scorer
        self.knowledge_scorer = None
        try:
            self.knowledge_scorer = KnowledgeScorer(
                cfg.get("knowledge_scorer", {}), memory, container, logger
            )
        except Exception as e:
            _logger.warning(f"KnowledgeScorer unavailable, falling back: {e}")

        self._evolution_log: List[Dict[str, Any]] = []

        self.use_arena = cfg.get("use_arena", True)
        self.sp_cfg = self.cfg.get("self_play", {"n_mc": 4, "batch": 6})

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
        self.report(
            {
                "event": "input",
                "agent": self.name,
                "docs_count": len(documents),
            }
        )

        # Process each document
        for di, paper in enumerate(documents, start=1):
            section_focus = context.get("section_name")
            doc_id = paper.get("id")
            title = paper.get("title", "")
            structured_sections = (
                self.memory.document_sections.get_by_document(doc_id)
            )

            # 1) Create CaseBook + Goal for this paper

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
                    "meta": build_paper_goal_meta(
                        title, doc_id, domains=self.cfg.get("domains", [])
                    ),
                }
            ).to_dict()

            # 2) Resolve sections with attributes
            sections_todo = self._resolve_sections_with_attributes(paper, context)
            results: List[Dict[str, Any]] = []
            abstract = self._get_abstract(structured_sections)
            for i, section in enumerate(structured_sections):
                section_text = section.section_text or ""
                case = self._create_section_case(casebook, paper, section, context)

                verify = await self._verify_and_improve(section, case, context)
                # Get a ranked corpus of related chat messages for this section
                corpus = self._get_corpus(section_text)

                # Persist results with attributes
                self._save_section_to_casebook(case, section, verify, context)

                results.append({
                    "section_name": section["section_name"],
                    "summary": verify["summary"],
                    "metrics": verify["metrics"],
                    "iterations": verify["iterations"],
                    "elapsed_ms": self._ms_since(st0),
                    "domain": section.get("domain"),
                })
                
                # Track strategy evolution
                self._track_strategy_evolution(case, verify["iterations"])
    

                buffers = self.memory.selfplay.ensure_buffer( 
                    section_id=section["id"]
                )
                proposals = []
                for _ in range(self.sp_cfg["batch"]):
                    prompt = self.prompt_loader.load_prompt(
                        "propose_task",
                        {
                            "section_text": section_text[:4000],
                            "fewshot": buffers.sample(
                                k=3
                            ),  # prior self-generated tasks for diversity
                            "task_type": self._choose_task_type(),  # 'deduce'|'abduce'|'induce'
                        },
                    )
                    raw_task = await self.call_llm(prompt, context)
                    task = self._validate_and_construct_task(
                        raw_task, section_text
                    )  # parse, check schema, safety
                    if task:
                        proposals.append(task)

                scored_props = []
                for task in proposals:
                    successes = []
                    logs = []
                    for _ in range(self.sp_cfg["n_mc"]):
                        s, detail = await solve_once(task)
                        successes.append(s)
                        logs.append(detail)
                    rbar = sum(successes) / max(1, len(successes))
                    r_propose = (
                        0.0 if rbar in (0.0, 1.0) else (1.0 - rbar)
                    )  # AZR Eq. (4)
                    scored_props.append(
                        {
                            "task": task,
                            "rbar": rbar,
                            "r_propose": r_propose,
                            "mc": logs,
                        }
                    )

                # Keep top-K learnable tasks
                keep = sorted(
                    scored_props, key=lambda x: x["r_propose"], reverse=True
                )[: self.cfg.get("self_play_keep", 4)]
                buffers.extend(keep)  # update buffer


                txt = section_text.strip()
                if len(txt) < self.min_section_length:
                    continue

            results: List[Dict[str, Any]] = []

            # 4) Per-section loop
            for si, section in enumerate(sections_todo, start=1):
                st0 = self._t0()
                section_name = section["section_name"]
                section_text = section["section_text"]

                # Build knowledge units (chat + paper)
                ku = self.builder.build(
                    chat_messages=section.get("corpus"),
                    paper_text=section_text,
                    conversation_id=context.get("conversation_id"),
                    context={**context, "section_name": section_name},
                )

                if self.use_arena:
                    # BEFORE: baseline = await self._baseline_summary(...)
                    # Instead, run the arena and use the winner as your “baseline”:
                    arena = await self._self_play_tournament(
                        paper, section, context
                    )
                    winner = arena["winner"]  # has .text and .score
                    baseline = winner["text"]

                    # (Optional) persist full arena for evidence & reuse:
                    self._persist_arena(casebook, paper, section, arena)
                else:
                    # Baseline summary
                    baseline = await self._baseline_summary(
                        paper, section, section.get("corpus"), context
                    )

                # Verify & improve (few iterations)
                verify = await self._verify_and_improve(
                    baseline, paper, section, context
                )

                # Persist artifacts to casebook (DSPy style)
                case = self._save_section_to_casebook(
                    casebook=casebook,
                    goal_id=paper_goal["id"],
                    doc_id=str(doc_id),
                    section_name=section_name,
                    section_text=section_text,
                    result={
                        "initial_draft": {
                            "title": section_name,
                            "body": baseline,
                        },
                        "refined_draft": {
                            "title": section_name,
                            "body": verify["summary"],
                        },
                        "verification_report": {
                            "scores": verify["metrics"],
                            "iterations": verify["iterations"],
                        },
                        "final_validation": {
                            "scores": verify["metrics"],
                            "passed": verify["metrics"]["overall"]
                            >= self.strategy.verification_threshold,
                        },
                        "passed": verify["metrics"]["overall"]
                        >= self.strategy.verification_threshold,
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
                    case_id=case.id,
                    baseline=baseline,
                    improved=verify["summary"],
                    metrics=verify["metrics"],
                    context=context,
                )

                results.append(
                    {
                        "section_name": section_name,
                        "summary": verify["summary"],
                        "metrics": verify["metrics"],
                        "iterations": verify["iterations"],
                        "elapsed_ms": self._ms_since(st0),
                    }
                )
                processed_sections.append(section)

            # Done
            out = {
                "paper_id": doc_id,
                "title": title,
                "results": results,
                "strategy": vars(self.strategy),
                "elapsed_ms": self._ms_since(t_run),
            }
            self.logger.log("LfL_Paper_Run_Complete", out)

            # After building `out` (per-paper results), add longitudinal analysis
            longitudinal = self._collect_longitudinal_metrics()
            evidence_md = self._generate_evidence_report(longitudinal)

            # Log a compact summary for telemetry
            self.logger.log(
                "LearningEvidence",
                {
                    "score_improvement_pct": longitudinal.get(
                        "score_improvement_pct"
                    ),
                    "iteration_reduction_pct": longitudinal.get(
                        "iteration_reduction_pct"
                    ),
                    "strategy_evolution_rate": longitudinal.get(
                        "strategy_evolution_rate"
                    ),
                    "total_papers": longitudinal.get("total_papers"),
                },
            )

            # Expose to downstream agents or UI
            out["longitudinal_metrics"] = longitudinal
            out["evidence_report_md"] = evidence_md

        return {**context, **out}

    # ---------------- section/prep  ----------------
    def _resolve_sections_with_attributes(self, paper: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve sections using attribute-based grouping"""
        doc_id = paper.get("id") or paper.get("doc_id")
        
        # Check if we have structured sections
        structured_sections = self.memory.document_sections.get_by_document(doc_id)
        
        if structured_sections:
            sections = []
            for sec in structured_sections:
                # Convert to attribute-based section
                sections.append({
                    "section_name": sec.section_name,
                    "section_text": sec.section_text,
                    "section_id": sec.id,
                    "order_index": getattr(sec, "order_index", None),
                    "attributes": {
                        "paper_id": doc_id,
                        "section_name": sec.section_name,
                        "section_index": getattr(sec, "order_index", 0),
                        "case_kind": "summary"
                    }
                })
            return sections
        
        # Fallback to abstract section
        return [{
            "section_name": "Abstract",
            "section_text": f"{paper.get('title','').strip()}\n\n{paper.get('abstract','').strip()}",
            "attributes": {
                "paper_id": doc_id,
                "section_name": "Abstract",
                "section_index": 0,
                "case_kind": "summary"
            }
        }]


    def _create_section_case(self, casebook: CaseBookORM, paper: Dict[str, Any], 
                            section: DocumentSectionORM, context: Dict[str, Any]) -> CaseORM:
        """Create a case with attributes for this section"""
        # Create the case
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=context.get("goal_id"),
            prompt_text=f"Section: {section['section_name']}",
            agent_name=self.name,
            meta={"type": "section_case"}
        )

        self.memory.casebooks.set_case_attr(case.id, "section_name", value_text=str(section.section_name))
        self.memory.casebooks.set_case_attr(case.id, "section_id", value_text=str(section.id)) 
        self.memory.casebooks.set_case_attr(case.id, "scorable_id", value_text=str(section.id)) 
        self.memory.casebooks.set_case_attr(case.id, "scorable_type", value_text=ScorableType.DOCUMENT_SECTION)
        
        return case

    def _track_strategy_evolution(self, case: CaseORM, iterations: List[Dict[str, Any]]):
        """Track strategy evolution using attributes"""
        if len(iterations) < 2:
            return

        gains = [iterations[i]["score"] - iterations[i-1]["score"] for i in range(1, len(iterations))]
        avg_gain = sum(gains)/len(gains) if gains else 0.0
        
        # Save evolution metrics as attributes
        self.memory.casebooks.set_case_attr(
            case.id, 
            "strategy_evolution", 
            value_json={
                "initial_strategy": {
                    "verification_threshold": self.strategy.verification_threshold,
                    "skeptic_weight": self.strategy.skeptic_weight,
                    "editor_weight": self.strategy.editor_weight,
                    "risk_weight": self.strategy.risk_weight
                },
                "avg_gain": avg_gain,
                "iterations": len(iterations)
            }
        )

    def _perform_cross_domain_learning(self, context: Dict[str, Any]):
        """Find the best patterns across all casebooks and content types"""
        # Get best cases across all casebooks
        best_cases = self.memory.casebooks.get_best_by_attrs(
            casebook_id=None,  # All casebooks
            group_keys=["case_kind", "domain"],
            score_weights={"knowledge_score": 0.8, "verification_score": 0.2},
            role="refined_draft",
            limit_per_group=3
        )
        
        # Analyze patterns across domains
        domain_patterns = {}
        for case in best_cases:
            domain = case["attributes"].get("domain", "general")
            pattern = self._extract_verification_pattern(case["text"])
            
            if domain not in domain_patterns:
                domain_patterns[domain] = []
            domain_patterns[domain].append(pattern)
        
        # Update strategy based on cross-domain patterns
        self._update_strategy_from_patterns(domain_patterns)
        
        # Log the cross-domain learning
        self.logger.log("CrossDomainLearning", {
            "domains": list(domain_patterns.keys()),
            "pattern_count": sum(len(p) for p in domain_patterns.values())
        })

    def _extract_verification_pattern(self, text: str) -> Dict[str, Any]:
        """Extract verification pattern from text"""
        # Simple pattern extraction (in production, use your NER/domain classifiers)
        patterns = {
            "checks": [],
            "common_phrases": [],
            "structure": []
        }
        
        # Check for common verification phrases
        verification_phrases = [
            "check if", "verify that", "validate", "ensure", "confirm", 
            "cross-reference", "compare against", "test case"
        ]
        
        for phrase in verification_phrases:
            if phrase in text.lower():
                patterns["checks"].append(phrase)
        
        # Analyze structure
        lines = text.split('\n')
        if len(lines) > 3:
            patterns["structure"].append("multi_paragraph")
        if "```" in text:
            patterns["structure"].append("code_example")
        if any(line.strip().startswith("-") for line in lines):
            patterns["structure"].append("bullet_points")
        
        return patterns

    def _update_strategy_from_patterns(self, domain_patterns: Dict[str, List[Dict[str, Any]]]):
        """Update strategy based on cross-domain patterns"""
        # Simple implementation - in production, use more sophisticated pattern matching
        for domain, patterns in domain_patterns.items():
            # Count common verification phrases
            phrase_counts = {}
            for pattern in patterns:
                for check in pattern["checks"]:
                    phrase_counts[check] = phrase_counts.get(check, 0) + 1
            
            # If "check if" is common in this domain, increase skeptic weight
            if phrase_counts.get("check if", 0) > len(patterns) * 0.7:
                self.strategy.skeptic_weight = min(0.60, self.strategy.skeptic_weight + 0.02)
                self.strategy.editor_weight = max(0.20, self.strategy.editor_weight - 0.01)
                self.strategy.risk_weight = max(0.20, self.strategy.risk_weight - 0.01)
                self.strategy.version += 1
                
                self.logger.log("DomainStrategyUpdate", {
                    "domain": domain,
                    "skeptic_weight": self.strategy.skeptic_weight,
                    "editor_weight": self.strategy.editor_weight,
                    "risk_weight": self.strategy.risk_weight,
                    "trigger": "high_check_if_usage"
                })
    def _get_abstract(self, paper_section) -> str:
        for section in paper_section:
            if section.section_name.lower() == "abstract":
                return section.section_text.strip()
        return ""

    def solve_once(
        self, task, context, section_text
    ) -> Tuple[int, Dict[str, Any]]:
        solve_prompt = self.prompt_loader.load_prompt(
            "solve_task", {"task": task}
        )
        ans = self.call_llm(solve_prompt, context)
        ok, details = self._verify_answer(
            task, ans, section_text
        )  # NLI/citation/consistency
        return int(ok), {"ans": ans, **details}

    async def _baseline_summary(
        self,
        paper: Dict[str, Any],
        section: Dict[str, Any],
        critical_msgs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> str:
        merged_context = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "focus_section": section.get("section_name"),
            "focus_text": section.get("section_text", "")[:5000],
            "hints": "\n".join(
                m.get("text", "")
                for m in (critical_msgs[:6] if critical_msgs else [])
            ),
            **context,
        }
        prompt = self.prompt_loader.from_file(
            "baseline_summary", self.cfg, merged_context
        )
        return self.call_llm(prompt, merged_context)

    async def _verify_and_improve(
        self,
        summary: str,
        paper: Dict[str, Any],
        section: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        max_iter = self.max_refinements
        current = summary
        iters: List[Dict[str, Any]] = []

        for i in range(1, max_iter + 1):
            metrics = self._score_summary(current, paper, section, context)
            iters.append(
                {
                    "iteration": i,
                    "score": metrics["overall"],
                    "metrics": metrics,
                }
            )

            if metrics["overall"] >= self.strategy.verification_threshold:
                break

            merged_context = {
                "title": paper.get("title", ""),
                "section_name": section.get("section_name"),
                "section_text": section.get("section_text", "")[:6000],
                "current_summary": current,
                "skeptic_weight": self.strategy.skeptic_weight,
                "editor_weight": self.strategy.editor_weight,
                "risk_weight": self.strategy.risk_weight,
                "weaknesses": json.dumps(
                    metrics.get("weaknesses", []), ensure_ascii=False
                ),
                **context,
            }

            improve_prompt = self.prompt_loader.from_file(
                "improve_summary", self.cfg, merged_context
            )
            current = self.call_llm(improve_prompt, context)

        # Meta-adapt *between* sections/papers
        self._evolve_strategy(iters, context)
        return {"summary": current, "metrics": metrics, "iterations": iters}

    # ---------------- scoring (two-head if available) ----------------

    def _normalize01(self, x, lo=0.0, hi=1.0):
        if x is None:
            return 0.0
        try:
            return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))
        except Exception:
            return 0.0

    def _length_penalty(self, text: str, target=(60, 600)):
        n = len(text or "")
        lo, hi = target
        if n <= 0:
            return 0.0
        if n < lo:
            return max(0.0, 1.0 - (lo - n) / lo) * 0.8
        if n > hi:
            return max(0.0, 1.0 - (n - hi) / hi) * 0.8
        return 1.0

    def _novelty_vs_corpus(self, answer: str, corpus_items: list) -> float:
        # Simple novelty: 1 - max semantic sim against top-k chat texts
        try:
            emb = self.memory.embedding
            a = emb.get_or_create(answer or "")
            mx = 0.0
            for it in corpus_items[:10]:
                b = emb.get_or_create(it.get("assistant_text", ""))
                num = sum(x * y for x, y in zip(a, b))
                na = (sum(x * x for x in a)) ** 0.5 or 1.0
                nb = (sum(y * y for y in b)) ** 0.5 or 1.0
                sim = max(0.0, min(1.0, num / (na * nb)))
                if sim > mx:
                    mx = sim
            return max(0.0, 1.0 - mx)
        except Exception:
            return 0.0

    def _solver_reward(
        self, task, answer_json, section_text, corpus_items
    ) -> dict:
        # 1) verifier
        ok, vinfo = self._verify_answer(task, answer_json, section_text)
        v = 1.0 if ok else 0.0

        # 2) knowledge dims (you already have _score_summary)
        ans_txt = (answer_json or {}).get("answer", "")
        dims = self._score_summary(
            ans_txt,
            {"title": "", "abstract": ""},
            {"section_text": section_text},
            {},
        )
        k = self._normalize01(dims.get("knowledge_score", 0.0))
        c = self._normalize01(dims.get("clarity", 0.0))
        g = self._normalize01(dims.get("grounding", 0.0))

        # 3) shaping
        len_pen = self._length_penalty(ans_txt)
        novelty = self._novelty_vs_corpus(ans_txt, corpus_items)

        eps, α, β, γ, δ = 0.05, 0.7, 0.15, 0.10, 0.05
        r = v * (eps + α * k + β * c + γ * g + δ * novelty) * len_pen
        return {
            "reward": float(round(r, 6)),
            "verify": vinfo,
            "dims": {
                "k": k,
                "c": c,
                "g": g,
                "novelty": novelty,
                "len_pen": len_pen,
            },
        }

    def _update_k_baseline(self, k_value: float):
        if not hasattr(self, "_k_baseline"):
            self._k_baseline = 0.55
        alpha = 0.1
        self._k_baseline = (1 - alpha) * self._k_baseline + alpha * float(
            k_value
        )
        return self._k_baseline

    def _proposer_reward(self, mc_rewards: list[dict]) -> dict:
        # mc_rewards: list of {"reward": R_solve, "dims": {...}} from _solver_reward
        if not mc_rewards:
            return {
                "rbar": 0.0,
                "kbar": 0.0,
                "R": 0.0,
                "L": 0.0,
                "k_base": self._k_baseline,
            }
        passes = [1.0 if r["reward"] > 0 else 0.0 for r in mc_rewards]
        ks = [r["dims"]["k"] for r in mc_rewards]
        rbar = sum(passes) / len(passes)
        kbar = sum(ks) / len(ks)
        L = 1.0 - abs(2 * rbar - 1.0)  # peak at 0.5
        k_base = self._update_k_baseline(kbar)
        lam = 0.7
        R = L * (lam * kbar + (1 - lam) * (kbar - k_base))
        return {
            "rbar": rbar,
            "kbar": kbar,
            "R": float(round(R, 6)),
            "L": L,
            "k_base": k_base,
        }

    async def _get_corpus(self, section_text: str) -> List[Dict[str, Any]]:
        corpus_search = self.chat_corpus(
            section_text,
            k=self.cfg.get("chat_corpus_k", 60),
            weights={  # optional; overrides defaults
                "semantic": 0.6,
                "entity": 0.25,
                "domain": 0.15,
            },
            include_text=True,  # you likely want the text downstream
        )
        # make sure to score/analyze the one we included, note typically this will be already done
        await self.annotate.run(context={"scorables": corpus_search.get("items", [])})
        await self.analyze.run(context={"chats": corpus_search.get("items", [])})

        return corpus_search.get("items", [])

    def _score_summary(
        self,
        text: str,
        paper: Dict[str, Any],
        section: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.knowledge_scorer:
            goal_text = (
                f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}"
            )
            p, comps = self.knowledge_scorer.model.predict(
                goal_text,
                text,
                meta={"text_len_norm": min(1.0, len(text) / 2000.0)},
                return_components=True,
            )
            knowledge = float(comps.get("probability", p))
            clarity, grounding = self._rubric_dims(
                text, section.get("section_text", "")
            )
            overall = 0.6 * knowledge + 0.25 * clarity + 0.15 * grounding
            weaknesses = self._weaknesses(
                text, section.get("section_text", "")
            )
            return {
                "overall": overall,
                "knowledge_score": knowledge,
                "clarity": clarity,
                "grounding": grounding,
                "weaknesses": weaknesses,
            }
        else:
            clarity, grounding = self._rubric_dims(
                text, section.get("section_text", "")
            )
            knowledge = 0.5 * clarity + 0.5 * grounding
            overall = 0.6 * knowledge + 0.25 * clarity + 0.15 * grounding
            weaknesses = self._weaknesses(
                text, section.get("section_text", "")
            )
            return {
                "overall": overall,
                "knowledge_score": knowledge,
                "clarity": clarity,
                "grounding": grounding,
                "weaknesses": weaknesses,
            }

    def _rubric_dims(self, text: str, ref: str) -> Tuple[float, float]:
        import re

        sents = [s for s in re.split(r"[.!?]\s+", text.strip()) if s]
        avg_len = sum(len(s.split()) for s in sents) / max(1, len(sents))
        clarity = max(0.0, min(1.0, 1.1 - (abs(avg_len - 22) / 22)))

        def toks(t):
            return set(re.findall(r"\b\w+\b", t.lower()))

        inter = len(toks(text) & toks(ref))
        grounding = max(0.0, min(1.0, inter / max(30, len(toks(ref)) or 1)))
        return clarity, grounding

    def _weaknesses(self, summary: str, ref: str) -> List[str]:
        out = []
        if len(summary) < 400:
            out.append("too short / thin detail")
        if "we propose" in ref.lower() and "we propose" not in summary.lower():
            out.append("misses core claim language")
        if summary.count("(") != summary.count(")"):
            out.append("formatting/parens issues")
        return out

    # ---------------- meta-learning ----------------

    def _evolve_strategy(
        self, iters: List[Dict[str, Any]], context: Optional[Dict[str, Any]]
    ):
        """
        Track iteration-to-iteration gains and evolve strategy.
        Also records an audit trail (old -> new) for evidence.
        """
        if len(iters) < 2:
            return

        # Compute average gain across iterations
        gains = [
            iters[i]["score"] - iters[i - 1]["score"]
            for i in range(1, len(iters))
        ]
        avg_gain = sum(gains) / len(gains) if gains else 0.0

        # Snapshot before
        old_strategy = {
            "verification_threshold": self.strategy.verification_threshold,
            "skeptic_weight": self.strategy.skeptic_weight,
            "editor_weight": self.strategy.editor_weight,
            "risk_weight": self.strategy.risk_weight,
            "version": self.strategy.version,
        }

        changed = False
        change_amount = (
            0.01  # default small nudge (only used in logs if needed)
        )
        if avg_gain < self.cfg.get("min_gain", 0.01):
            # scale change for very-low-gain regimes
            change_amount = 0.06 if avg_gain < 0.005 else 0.03
            self.strategy.skeptic_weight = min(
                0.60, self.strategy.skeptic_weight + change_amount
            )
            self.strategy.editor_weight = max(
                0.20, self.strategy.editor_weight - change_amount / 2
            )
            self.strategy.risk_weight = max(
                0.20, self.strategy.risk_weight - change_amount / 2
            )
            changed = True
        elif avg_gain > self.cfg.get("high_gain", 0.03):
            self.strategy.verification_threshold = max(
                0.80, self.strategy.verification_threshold - 0.01
            )
            changed = True

        if changed:
            self.strategy.version += 1

            new_strategy = {
                "verification_threshold": self.strategy.verification_threshold,
                "skeptic_weight": self.strategy.skeptic_weight,
                "editor_weight": self.strategy.editor_weight,
                "risk_weight": self.strategy.risk_weight,
                "version": self.strategy.version,
            }

            # Persist audit trail in-memory and log
            event = {
                "avg_gain": round(avg_gain, 4),
                "change_amount": change_amount,
                "old": old_strategy,
                "new": new_strategy,
                "iteration_count": len(iters),
                "timestamp": time.time(),
            }
            self._evolution_log.append(event)
            _logger.info(f"LfL_Strategy_Evolved: {event}")

            # If a context dict is provided, also push into it (handy for downstream agents)
            context.setdefault("strategy_evolution", []).append(event)

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
            if extra:
                base.update(extra)
            return base

        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=section_text,
            role="section_text",
            meta=_smeta(),
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=self._dumps_safe(result.get("initial_draft", {})),
            role="initial_draft",
            meta=_smeta(),
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=self._dumps_safe(result.get("refined_draft", {})),
            role="refined_draft",
            meta=_smeta(
                {
                    "refinement_iterations": result.get(
                        "refinement_iterations", 0
                    )
                }
            ),
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=self._dumps_safe(result.get("verification_report", {})),
            role="verification_report",
            meta=_smeta(),
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=self._dumps_safe(result.get("final_validation", {})),
            role="final_validation",
            meta=_smeta(),
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=self._dumps_safe(
                {
                    "passed": result.get("passed", False),
                    "refinement_iterations": result.get(
                        "refinement_iterations", 0
                    ),
                    "final_scores": (
                        result.get("final_validation", {}) or {}
                    ).get("scores", {}),
                }
            ),
            role="metrics",
            meta=_smeta(),
        )
        return case

    # ---------------- DPO-lite pair persistence ----------------

    def _persist_pairs(
        self,
        paper_id: Any,
        case_id: int,
        baseline: str,
        improved: str,
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ):
        try:
            if not paper_id:
                paper_id = f"paper:{hash((baseline, improved))}"
            self.memory.casebooks.add_scorable(
                case_id=case_id,
                role="knowledge_pair_positive",
                text=improved,
                pipeline_run_id=context.get("pipeline_run_id"),
                meta={
                    "verification_score": metrics.get("overall", 0.0),
                    "knowledge_score": metrics.get("knowledge_score", 0.0),
                    "strategy_version": self.strategy.version,
                },
            )
            if (
                metrics.get("overall", 0.0)
                >= self.strategy.verification_threshold
            ):
                self.memory.casebooks.add_scorable(
                    case_id=case_id,
                    role="knowledge_pair_negative",
                    text=baseline,
                    meta={
                        "verification_score": max(
                            0.0, metrics.get("overall", 0.0) - 0.15
                        ),
                        "knowledge_score": max(
                            0.0, metrics.get("knowledge_score", 0.0) * 0.7
                        ),
                        "strategy_version": self.strategy.version,
                    },
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
            return json.dumps(
                {"_warning": "failed_to_dump", "repr": repr(obj)}
            )

    def _percent_change(self, start: float, end: float) -> float:
        try:
            if start is None or end is None or start == 0:
                return 0.0
            return (end - start) / abs(start) * 100.0
        except Exception:
            return 0.0

    def _collect_longitudinal_metrics(self) -> Dict[str, Any]:
        """
        Aggregate verification scores/iterations across prior papers to show cross-episode improvement.
        Reads back the persisted 'metrics' scorables saved in _save_section_to_casebook.
        Designed to fail gracefully if stores/APIs differ.
        """
        out = {
            "total_papers": 0,
            "verification_scores": [],
            "iteration_counts": [],
            "avg_verification_score": 0.0,
            "avg_iterations": 0.0,
            "score_improvement_pct": 0.0,
            "iteration_reduction_pct": 0.0,
            "strategy_versions": [],
            "strategy_evolution_rate": 0.0,
        }

        try:
            casebooks = self.memory.casebooks.get_casebooks_by_tag(
                self.casebook_action
            )
            for cb in casebooks:
                # For each case, gather the 'metrics' scorable(s) you saved
                # You saved a dictionary with fields:
                #   passed, refinement_iterations, final_scores: { overall: ... }
                cases = self.memory.casebooks.get_cases_for_casebook(cb.id)
                for case in cases:
                    scs = self.memory.casebooks.list_scorables(case.id)
                    for s in scs:
                        try:
                            payload = (
                                json.loads(s.text)
                                if isinstance(s.text, str)
                                else (s.text or {})
                            )
                            final_scores = payload.get("final_scores") or {}
                            overall = final_scores.get("overall")
                            iters = payload.get("refinement_iterations")
                            if overall is not None:
                                out["verification_scores"].append(
                                    float(overall)
                                )
                            if iters is not None:
                                out["iteration_counts"].append(int(iters))
                        except Exception:
                            continue

            vs = out["verification_scores"]
            it = out["iteration_counts"]
            out["total_papers"] = len(vs)

            if vs:
                out["avg_verification_score"] = sum(vs) / len(vs)
            if it:
                out["avg_iterations"] = sum(it) / len(it)

            if len(vs) >= 2:
                out["score_improvement_pct"] = self._percent_change(
                    vs[0], vs[-1]
                )
            if len(it) >= 2:
                out["iteration_reduction_pct"] = (
                    self._percent_change(it[0], it[-1]) * -1.0
                )  # fewer is better

            # Strategy evolution (from in-memory log)
            versions = (
                [e["new"]["version"] for e in self._evolution_log]
                if self._evolution_log
                else []
            )
            out["strategy_versions"] = versions
            if versions:
                out["strategy_evolution_rate"] = len(set(versions)) / max(
                    1, len(versions)
                )

            return out
        except Exception as e:
            try:
                self.logger.log("LfL_Longitudinal_Failed", {"err": str(e)})
            except Exception:
                pass
            return out

    def _generate_evidence_report(self, longitudinal: Dict[str, Any]) -> str:
        """
        Compact, blog-ready summary. Returns empty string if not enough data yet.
        """
        if not longitudinal or longitudinal.get("total_papers", 0) < 3:
            return ""

        score_trend = longitudinal.get("score_improvement_pct", 0.0)
        iter_trend = longitudinal.get("iteration_reduction_pct", 0.0)
        arrow_score = "↑" if score_trend > 0 else "↓"
        arrow_iter = "↓" if iter_trend > 0 else "↑"

        vs = longitudinal.get("verification_scores", [])
        it = longitudinal.get("iteration_counts", [])

        # Build a short Markdown report
        lines = []
        lines.append("## Learning from Learning: Evidence Report")
        lines.append("")
        lines.append(
            f"- **Total papers processed**: {longitudinal.get('total_papers', 0)}"
        )
        lines.append(
            f"- **Verification score trend**: {score_trend:.1f}% {arrow_score}"
        )
        lines.append(
            f"- **Average iterations trend**: {iter_trend:.1f}% {arrow_iter}"
        )
        lines.append(
            f"- **Strategy evolution events**: {max(0, len(set(longitudinal.get('strategy_versions', []))) - 1)}"
        )
        lines.append("")
        if vs and it:
            lines.append("### Snapshot")
            lines.append(
                f"- First paper: score={vs[0]:.2f}, iterations={it[0] if it else 'n/a'}"
            )
            lines.append(
                f"- Latest paper: score={vs[-1]:.2f}, iterations={it[-1] if it else 'n/a'}"
            )
        return "\n".join(lines)

    def _collect_section_candidates(self, paper, section, context):
        section_name = section["section_name"]
        section_text = section["section_text"]

        candidates = []

        # 1) External agent drafts already in context (assume they stored under context["agent_drafts"][section_name])
        for d in context.get("agent_drafts", {}).get(section_name, []) or []:
            candidates.append(
                {
                    "origin": d.get("agent_name", "external_agent"),
                    "variant": d.get("variant", "v1"),
                    "text": d.get("text", ""),
                    "meta": {"source": "agent_pool", **(d.get("meta") or {})},
                }
            )

        # 2) Chat corpus retrieval → extract usable snippets (assistant_text)
        corpus = self.chat_corpus(
            section_text,
            k=self.cfg.get("chat_corpus_k", 60),
            include_text=True,
        )
        for it in corpus.get("items", [])[
            : self.cfg.get("max_corpus_candidates", 8)
        ]:
            t = it.get("assistant_text", "").strip()
            if not t:
                continue
            candidates.append(
                {
                    "origin": "chat_corpus",
                    "variant": f"c{it['id']}",
                    "text": t,
                    "meta": {
                        "scores": it.get("scores", {}),
                        "source": "chat_corpus",
                        "turn_id": it["id"],
                    },
                }
            )

        # 3) Past winners in CaseBook (same paper/section)
        try:
            past = (
                self.memory.casebooks.find_best_for_section(
                    paper_id=str(paper.get("id") or paper.get("doc_id")),
                    section_name=section_name,
                    limit=self.cfg.get("max_past_candidates", 4),
                )
                or []
            )
            for p in past:
                candidates.append(
                    {
                        "origin": "casebook_winner",
                        "variant": f"cb_{p.id}",
                        "text": p.text or "",
                        "meta": {
                            "source": "casebook",
                            "score": p.meta.get("verification_score"),
                        },
                    }
                )
        except Exception:
            pass

        # 4) LfL’s own first baseline (seed)
        base = {
            "origin": "lfl_baseline",
            "variant": "baseline",
            "text": "",
            "meta": {"source": "lfl"},
        }
        try:
            base["text"] = self.prompt_loader.load_prompt(
                "section_seed",
                {
                    "focus": section_name,
                    "paper_title": paper.get("title", ""),
                    "section_text": section_text[:6000],
                },
            )
        except Exception:
            base["text"] = section_text[:800]
        candidates.append(base)

        # Dedup by normalized text
        seen = set()
        out = []
        for c in candidates:
            key = " ".join(c["text"].split()).lower()
            if len(key) < 40:  # drop super-short
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _score_candidate(self, text: str, section_text: str) -> dict:
        # Use your scorer (knowledge, clarity, grounding) and a simple verifier gate
        dims = self._score_summary(
            text,
            {"title": "", "abstract": ""},
            {"section_text": section_text},
            {},
        )
        k = float(dims.get("knowledge_score", 0.0))
        c = float(dims.get("clarity", 0.0))
        g = float(dims.get("grounding", 0.0))
        overall = 0.6 * k + 0.25 * c + 0.15 * g

        # Lightweight verifier gate: require minimal grounding & structure
        verified = (g >= 0.45) and (
            len(text) >= self.cfg.get("min_verified_len", 250)
        )
        return {
            "k": k,
            "c": c,
            "g": g,
            "overall": overall,
            "verified": bool(verified),
        }

    async def _self_play_tournament(self, paper, section, context) -> dict:
        section_text = section["section_text"]
        pool = self._collect_section_candidates(paper, section, context)

        # 1) initial scoring
        scored = []
        for cand in pool:
            s = self._score_candidate(cand["text"], section_text)
            cand["score"] = s
            scored.append(cand)
        scored.sort(
            key=lambda x: (x["score"]["verified"], x["score"]["overall"]),
            reverse=True,
        )

        beam = scored[: self.cfg.get("beam_width", 5)]

        # 2) a few self-play improvement rounds
        iters = []
        rounds = self.cfg.get("self_play_rounds", 2)
        for r in range(rounds):
            new_beam = []
            for cand in beam:
                improve_prompt = self.prompt_loader.load_prompt(
                    "improve_summary",
                    {
                        "title": paper.get("title", ""),
                        "section_name": section.get("section_name"),
                        "section_text": section_text[:6000],
                        "current_summary": cand["text"],
                        "skeptic_weight": self.strategy.skeptic_weight,
                        "editor_weight": self.strategy.editor_weight,
                        "risk_weight": self.strategy.risk_weight,
                        "weaknesses": json.dumps(
                            self._weaknesses(cand["text"], section_text),
                            ensure_ascii=False,
                        ),
                    },
                )
                improved = await self.call_llm(improve_prompt, context)
                s = self._score_candidate(improved, section_text)
                new_beam.append(
                    {
                        "origin": cand["origin"],
                        "variant": f"{cand['variant']}+r{r + 1}",
                        "text": improved,
                        "meta": cand.get("meta", {}),
                        "score": s,
                    }
                )
            # keep top beam
            new_beam.sort(
                key=lambda x: (x["score"]["verified"], x["score"]["overall"]),
                reverse=True,
            )
            beam = new_beam[: self.cfg.get("beam_width", 5)]
            iters.append(
                [
                    {
                        "variant": b["variant"],
                        "overall": b["score"]["overall"],
                        "k": b["score"]["k"],
                    }
                    for b in beam
                ]
            )

        winner = (
            beam[0]
            if beam
            else (scored[0] if scored else {"text": "", "score": {}})
        )
        return {
            "winner": winner,
            "beam": beam,
            "initial_pool": scored,
            "iterations": iters,
        }

    def _persist_arena(self, casebook, paper, section, arena): 
        # Save every candidate (initial pool), each round’s beam, and final winner
        def _meta(base=None, **kw):
            m = {
                "paper_id": str(paper.get("id") or paper.get("doc_id")),
                "section_name": section["section_name"],
                "type": "arena",
            }
            if base:
                m.update(base)
            m.update(kw)
            return m

        # Initial pool
        for c in arena["initial_pool"][: self.cfg.get("persist_pool_cap", 30)]:
            self.memory.casebooks.add_scorable(
                case_id=casebook.id,
                role="arena_candidate",
                text=c["text"],
                meta=_meta(
                    origin=c["origin"], variant=c["variant"], score=c["score"]
                ),
            )
        # Rounds
        for ri, beam in enumerate(
            arena["beam"][: self.cfg.get("persist_beam_cap", 10)]
        ):
            self.memory.casebooks.add_scorable(
                case_id=casebook.id,
                role="arena_beam",
                text=beam["text"],
                meta=_meta(
                    round=ri,
                    origin=beam["origin"],
                    variant=beam["variant"],
                    score=beam["score"],
                ),
            )
        # Winner
        w = arena["winner"]
        self.memory.casebooks.add_scorable(
            case_id=casebook.id,
            role="arena_winner",
            text=w["text"],
            meta=_meta(
                origin=w["origin"], variant=w["variant"], score=w["score"]
            ),
        )
