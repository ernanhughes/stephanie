# stephanie/agents/learning_from_learning.py
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.knowledge.chat_analyze import ChatAnalyzeAgent
from stephanie.agents.knowledge.chat_knowledge_builder import \
    ChatKnowledgeBuilder
from stephanie.agents.knowledge.conversation_filter import \
    ConversationFilterAgent
from stephanie.agents.knowledge.scorable_annotate import ScorableAnnotateAgent
from stephanie.models.casebook import CaseBookORM, CaseORM
from stephanie.scoring.scorable import ScorableType
from stephanie.scoring.scorer.knowledge_scorer import KnowledgeScorer
from stephanie.tools.chat_corpus_tool import build_chat_corpus_tool
from stephanie.utils.casebook_utils import generate_casebook_name
from stephanie.utils.json_sanitize import dumps_safe  # and/or sanitize
from stephanie.utils.paper_utils import (build_paper_goal_meta,
                                         build_paper_goal_text)
from stephanie.utils.agent_progress import AgentProgress

_logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    verification_threshold: float = 0.85
    skeptic_weight: float = 0.34
    editor_weight: float = 0.33
    risk_weight: float = 0.33
    version: int = 1

class _OriginRouter:
    def __init__(self):
        self._μ = {}      # running mean
        self._n = {}      # count

    def update(self, origin: str, reward: float):
        μ = self._μ.get(origin, 0.6)
        n = self._n.get(origin, 0)
        n2 = n + 1
        μ2 = (μ * n + reward) / n2
        self._μ[origin], self._n[origin] = μ2, n2

    def topk(self, origins: List[str], k: int) -> List[str]:
        scored = [(o, self._μ.get(o, 0.6)) for o in set(origins)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [o for o,_ in scored[:k]]


class LearningFromLearningAgent(BaseAgent):
    """
    Paper → sections → (arena|baseline) → verify/improve → persist (+ evidence).
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

        # Config
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

        # Chat corpus tool
        self.chat_corpus = build_chat_corpus_tool(
            memory=memory, container=container, cfg=cfg.get("chat_corpus", {})
        )

        # Optional two-head scorer
        self.knowledge_scorer = None
        try:
            self.knowledge_scorer = KnowledgeScorer(
                cfg.get("knowledge_scorer", {}), memory, container, logger
            )
        except Exception as e:
            _logger.warning(f"KnowledgeScorer unavailable, falling back: {e}")

        # Arena switch
        self.use_arena = bool(cfg.get("use_arena", True))

        self._origin_router = _OriginRouter()

        # Evolution log
        self._evolution_log: List[Dict[str, Any]] = []
        self.progress = AgentProgress(self, enable_sis=bool(cfg.get("progress", {}).get("enable_sis", True)))
        self.progress_attrs = bool(cfg.get("progress", {}).get("write_case_attrs", True))
        self.single_random_doc = bool(cfg.get("single_random_doc", True))


    # ---------------- public entry ----------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        t_run = self._t0()

        documents = context.get(self.input_key, []) or []

        # test single doc mode
        if self.single_random_doc:
            doc = self.memory.documents.get_random()
            documents = [doc.to_dict()] 

        self.report(
            {
                "event": "input",
                "agent": self.name,
                "docs_count": len(documents),
            }
        )

        out = {}
        pipeline_run_id = context.get("pipeline_run_id")
        for paper in documents:
            doc_id = paper.get("id") or paper.get("doc_id")
            title = paper.get("title", "")

            # Casebook + goal
            casebook_name = generate_casebook_name(self.casebook_action, title)
            casebook = self.memory.casebooks.ensure_casebook(
                name=casebook_name,
                pipeline_run_id=pipeline_run_id,
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

            # Resolve sections (dicts with attributes)
            sections_todo = self._resolve_sections_with_attributes(
                paper, context
            )
            self._progress_start_paper(paper, sections_todo)
            results: List[Dict[str, Any]] = []

            for si, section in enumerate(sections_todo, start=1):
                st0 = self._t0()
                section_name = section["section_name"]
                section_text = section["section_text"]

                self._progress_start_section(section, si)

                # Skip tiny sections
                if (
                    not (section_text or "").strip()
                    or len(section_text) < self.min_section_length
                ):
                    continue

                # Create case + attributes for this section
                case = self._create_section_case(
                    casebook, paper, section, {**context, "goal": paper_goal}
                )
                context_with_case = {**context, "case_id": case.id, "pipeline_run_id": context.get("pipeline_run_id")}


                # Retrieve corpus (+ annotate & analyze)
                self._progress_stage(section, "corpus:start")
                corpus_items = await self._get_corpus(section_text)
                self._progress_stage(section, "corpus:done", items=len(corpus_items or []))

                # Arena or baseline
                if self.use_arena:
                    arena =self._self_play_tournament(
                        paper, section, context_with_case
                    )
                    winner = arena["winner"]
                    baseline = winner["text"]
                    self._persist_arena(
                        case, paper, section, arena, context=context
                    )
                    self._progress_stage(section, "arena:done", winner_overall=round(float(winner.get("score", {}).get("overall", 0.0)), 3))
                else:
                    baseline = self._baseline_summary(
                        paper, section, corpus_items, context_with_case
                    )
                    self._progress_stage(section, "baseline:done")

                # Verify & improve
                self._progress_stage(section, "verify:start")
                verify = self._verify_and_improve(
                    baseline, paper, section, context_with_case
                )
                self._progress_stage(section, "verify:done", overall=round(float(verify["metrics"].get("overall", 0.0)), 3))

                # Persist artifacts
                self._progress_stage(section, "persist:start")
                saved_case = self._save_section_to_casebook(
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
                self._progress_stage(section, "persist:done")

                # Track strategy evolution per section
                self._track_strategy_evolution(
                    saved_case, verify["iterations"]
                )

                # Persist preference pairs
                self._persist_pairs(
                    case_id=saved_case.id,
                    baseline=baseline,
                    improved=verify["summary"],
                    metrics=verify["metrics"],
                    context=context,
                )
                section_metrics = {
                    "overall": verify["metrics"].get("overall", 0.0),
                    "refinement_iterations": len(verify["iterations"]),
                    "elapsed_ms": self._ms_since(st0),
                }
                self._progress_end_section(saved_case, section, section_metrics)
                self._validate_strategy_effectiveness(context)
                results.append(
                    {
                        "section_name": section_name,
                        "summary": verify["summary"],
                        "metrics": verify["metrics"],
                        "iterations": verify["iterations"],
                        "elapsed_ms": self._ms_since(st0),
                    }
                )

            # Per-paper output + longitudinal evidence
            paper_out = {
                "paper_id": doc_id,
                "title": title,
                "results": results,
                "strategy": vars(self.strategy),
                "elapsed_ms": self._ms_since(t_run),
            }
            self.logger.log("LfL_Paper_Run_Complete", paper_out)

            longitudinal = self._collect_longitudinal_metrics()
            cross = self._calculate_cross_episode_metrics()
            # Optional: attach A/B validation snapshot into cross so evidence report can render it
            ab_val = self._validate_strategy_effectiveness(context)
            if ab_val:
                cross["strategy_ab_validation"] = ab_val
            # Store for later use and build evidence once
            self._last_cross_episode = cross
            evidence_md = self._generate_evidence_report(longitudinal, cross)
            paper_out["cross_episode"] = cross
            paper_out["longitudinal_metrics"] = longitudinal
            paper_out["evidence_report_md"] = evidence_md
            out.update(paper_out)

        return {**context, **out}

    # ---------------- section resolving / attributes ----------------

    def _resolve_sections_with_attributes(
        self, paper: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Normalize sections to dicts with attributes.
        """
        doc_id = paper.get("id") or paper.get("doc_id")
        sections = self.memory.document_sections.get_by_document(doc_id) or []

        if sections:
            out = []
            for sec in sections:
                out.append(
                    {
                        "section_name": sec.section_name,
                        "section_text": sec.section_text or "",
                        "section_id": sec.id,
                        "order_index": getattr(sec, "order_index", None),
                        "attributes": {
                            "paper_id": str(doc_id),
                            "section_name": sec.section_name,
                            "section_index": getattr(sec, "order_index", 0),
                            "case_kind": "summary",
                        },
                    }
                )
            return out

        # Fallback single "Abstract" section
        return [
            {
                "section_name": "Abstract",
                "section_text": f"{paper.get('title', '').strip()}\n\n{paper.get('abstract', '').strip()}",
                "section_id": None,
                "order_index": 0,
                "attributes": {
                    "paper_id": str(doc_id),
                    "section_name": "Abstract",
                    "section_index": 0,
                    "case_kind": "summary",
                },
            }
        ]

    def _create_section_case(
        self,
        casebook: CaseBookORM,
        paper: Dict[str, Any],
        section: Dict[str, Any],
        context: Dict[str, Any],
    ) -> CaseORM:
        """
        Create a case and attach attributes for this section (universal casebook pattern).
        """
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=context.get("goal").get("id"),
            prompt_text=f"Section: {section['section_name']}",
            agent_name=self.name,
            meta={"type": "section_case"},
        )

        # Attributes
        self.memory.casebooks.set_case_attr(
            case.id,
            "paper_id",
            value_text=str(paper.get("id") or paper.get("doc_id")),
        )
        self.memory.casebooks.set_case_attr(
            case.id, "section_name", value_text=str(section["section_name"])
        )
        if section.get("section_id") is not None:
            self.memory.casebooks.set_case_attr(
                case.id, "section_id", value_text=str(section["section_id"])
            )
        if section.get("order_index") is not None:
            self.memory.casebooks.set_case_attr(
                case.id,
                "section_index",
                value_num=float(section.get("order_index") or 0),
            )
        self.memory.casebooks.set_case_attr(
            case.id, "case_kind", value_text="summary"
        )
        self.memory.casebooks.set_case_attr(
            case.id,
            "scorable_id",
            value_text=str(section.get("section_id") or ""),
        )
        self.memory.casebooks.set_case_attr(
            case.id,
            "scorable_type",
            value_text=str(ScorableType.DOCUMENT_SECTION),
        )

        return case

    # ---------------- corpus / baseline / verify ----------------

    async def _get_corpus(self, section_text: str) -> List[Dict[str, Any]]:
        try:
            corpus_search = self.chat_corpus(
                section_text,
                k=self.cfg.get("chat_corpus_k", 60),
                weights={"semantic": 0.6, "entity": 0.25, "domain": 0.15},
                include_text=True,
            )
            items = corpus_search.get("items", []) or []
            try:
                await self.annotate.run(context={"scorables": items})
                await self.analyze.run(context={"chats": items})
            except Exception as e:
                _logger.warning(f"Corpus annotate/analyze skipped: {e}")
            return items
        except Exception as e:
            _logger.warning(f"Chat corpus retrieval failed: {e}")
            return []

    def _verify_and_improve(
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

        self._evolve_strategy(iters, context)
        return {"summary": current, "metrics": metrics, "iterations": iters}

    # ---------------- arena ----------------

    def _score_candidate(self, text: str, section_text: str) -> dict:
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

    def _track_strategy_evolution(
        self,
        case: CaseORM,
        iterations: List[Dict[str, Any]],
    ) -> None:
        """
        Persist a compact audit trail of per-section strategy dynamics.
        - Computes average per-iteration gain
        - Records current strategy knobs
        - Stores as a case attribute (JSON) so it's queryable later
        """
        try:
            if not iterations or len(iterations) < 2:
                # Still record the current strategy so we can track across cases
                payload = {
                    "avg_gain": 0.0,
                    "iteration_count": len(iterations) if iterations else 0,
                    "strategy": {
                        "verification_threshold": self.strategy.verification_threshold,
                        "skeptic_weight": self.strategy.skeptic_weight,
                        "editor_weight": self.strategy.editor_weight,
                        "risk_weight": self.strategy.risk_weight,
                        "version": self.strategy.version,
                    },
                    "timestamp": time.time(),
                }
                self.memory.casebooks.set_case_attr(
                    case.id, "strategy_evolution", value_json=payload
                )
                return

            # Compute average gain across iterations
            gains = [
                iterations[i]["score"] - iterations[i - 1]["score"]
                for i in range(1, len(iterations))
                if "score" in iterations[i] and "score" in iterations[i - 1]
            ]
            avg_gain = (sum(gains) / len(gains)) if gains else 0.0

            payload = {
                "avg_gain": round(avg_gain, 6),
                "iteration_count": len(iterations),
                "strategy": {
                    "verification_threshold": self.strategy.verification_threshold,
                    "skeptic_weight": self.strategy.skeptic_weight,
                    "editor_weight": self.strategy.editor_weight,
                    "risk_weight": self.strategy.risk_weight,
                    "version": self.strategy.version,
                },
                "timestamp": time.time(),
            }

            # Persist as a JSON attribute on the case
            self.memory.casebooks.set_case_attr(
                case.id, "strategy_evolution", value_json=payload
            )

            # Also keep a simple in-memory log if you want longitudinal stats later
            self._evolution_log.append(
                {
                    "avg_gain": payload["avg_gain"],
                    "new": payload["strategy"],
                    "iteration_count": payload["iteration_count"],
                    "timestamp": payload["timestamp"],
                }
            )

            self.logger.log("LfL_Strategy_Track", payload)

        except Exception as e:
            _logger.warning(f"_track_strategy_evolution skipped: {e}")

    def _record_strategy_state(self, context: Optional[Dict[str, Any]], tag: str = "pre_change") -> None:
        """Record current strategy knobs so we can compare later."""
        payload = {
            "tag": tag,
            "verification_threshold": self.strategy.verification_threshold,
            "skeptic_weight": self.strategy.skeptic_weight,
            "editor_weight": self.strategy.editor_weight,
            "risk_weight": self.strategy.risk_weight,
            "version": self.strategy.version,
            "timestamp": time.time(),
        }
        try:
            # If a current case_id is available in context, attach it; otherwise, store on the pipeline run
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="strategy_state", text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"tag": tag}
                )
        except Exception:
            pass

    def _propose_strategy_changes(self, avg_gain: float) -> Strategy:
        """Return a *copy* of the current strategy with proposed adjustments applied."""
        proposed = Strategy(
            verification_threshold=self.strategy.verification_threshold,
            skeptic_weight=self.strategy.skeptic_weight,
            editor_weight=self.strategy.editor_weight,
            risk_weight=self.strategy.risk_weight,
            version=self.strategy.version,
        )
        change_amount = 0.06 if avg_gain < self.cfg.get("min_gain", 0.01) and avg_gain < 0.005 else 0.03
        if avg_gain < self.cfg.get("min_gain", 0.01):
            proposed.skeptic_weight = min(0.60, proposed.skeptic_weight + change_amount)
            proposed.editor_weight  = max(0.20, proposed.editor_weight - change_amount / 2)
            proposed.risk_weight    = max(0.20, proposed.risk_weight   - change_amount / 2)
        elif avg_gain > self.cfg.get("high_gain", 0.03):
            proposed.verification_threshold = max(0.80, proposed.verification_threshold - 0.01)
        return proposed

    def _record_strategy_test(self, context: Optional[Dict[str, Any]], old_strategy: Dict[str, Any],
                            new_strategy: Strategy, test_group: str, avg_gain: float) -> None:
        """Record that we entered A or B for upcoming work."""
        payload = {
            "test_group": test_group,
            "avg_gain": avg_gain,
            "old": old_strategy,
            "new": {
                "verification_threshold": new_strategy.verification_threshold,
                "skeptic_weight": new_strategy.skeptic_weight,
                "editor_weight": new_strategy.editor_weight,
                "risk_weight": new_strategy.risk_weight,
                "version": new_strategy.version,
            },
            "timestamp": time.time(),
        }
        try:
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="strategy_ab_enroll",
                    text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"group": test_group}
                )
        except Exception:
            pass


    def _collect_section_candidates(self, paper, section, context):
        name = section["section_name"]
        text = section["section_text"]
        cands = []

        # 1) external agent drafts supplied in context
        for d in context.get("agent_drafts", {}).get(name, []) or []:
            cands.append(
                {
                    "origin": d.get("agent_name", "external_agent"),
                    "variant": d.get("variant", "v1"),
                    "text": d.get("text", ""),
                    "meta": {"source": "agent_pool", **(d.get("meta") or {})},
                }
            )

        # 2) chat corpus snippets
        corpus = self.chat_corpus(
            text, k=self.cfg.get("chat_corpus_k", 60), include_text=True
        )
        for it in corpus.get("items", [])[
            : self.cfg.get("max_corpus_candidates", 8)
        ]:
            t = (it.get("assistant_text") or "").strip()
            if not t:
                continue
            cands.append(
                {
                    "origin": "chat_corpus",
                    "variant": f"c{it['id']}",
                    "text": t,
                    "meta": {
                        "scores": it.get("scores", {}),
                        "turn_id": it["id"],
                    },
                }
            )

        # 3) past winners
        try:
            past = (
                self.memory.casebooks.find_best_for_section(
                    paper_id=str(paper.get("id") or paper.get("doc_id")),
                    section_name=name,
                    limit=self.cfg.get("max_past_candidates", 4),
                )
                or []
            )
            for p in past:
                cands.append(
                    {
                        "origin": "casebook_winner",
                        "variant": f"cb_{p.id}",
                        "text": p.text or "",
                        "meta": {"score": p.meta.get("verification_score")},
                    }
                )
        except Exception:
            pass

        # 4) baseline seed (prompt template content or safe fallback)
        try:
            merged_context = {
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "focus_section": name,
                "section_text": text[:5000],
                **context,
            }
            seed = self.prompt_loader.from_file(
                "section_seed", self.cfg, merged_context
            )
        except Exception:
            seed = text[:800]
        cands.append(
            {
                "origin": "lfl_baseline",
                "variant": "baseline",
                "text": seed,
                "meta": {"source": "lfl"},
            }
        )

        # Dedup
        seen, out = set(), []
        for c in cands:
            key = " ".join((c["text"] or "").split()).lower()
            if len(key) < 40:  # too short
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _self_play_tournament(self, paper, section, context) -> dict:
        section_text = section["section_text"]

        # 1) Collect & score initial pool
        pool = self._collect_section_candidates(paper, section, context)
        scored = []
        for cand in pool:
            s = self._score_candidate(cand["text"], section_text)
            cand["score"] = s
            scored.append(cand)
        scored.sort(key=lambda x: (x["score"]["verified"], x["score"]["overall"]), reverse=True)

        beam_w = int(self.cfg.get("beam_width", 5))
        beam = scored[:beam_w]
        iters = []

        # Config knobs
        max_rounds = int(self.cfg.get("self_play_rounds", 2))
        plateau_eps = float(self.cfg.get("self_play_plateau_eps", 0.005))
        min_marg = float(self.cfg.get("min_marginal_reward_per_ktok", 0.05))

        # Helpers (local, no external deps)
        def _est_tokens(txt: str) -> int:
            return max(1, int(len(txt or "") / 4))  # ~4 chars/token

        def _marginal(prev_best: float, curr_best: float, prev_toks: int, curr_toks: int) -> float:
            dr = curr_best - prev_best
            dt = max(1, curr_toks - prev_toks)
            return (dr / dt) * 1000.0  # reward per 1k tokens

        best_hist = []
        prev_best = beam[0]["score"]["overall"] if beam else 0.0
        prev_tokens = _est_tokens(beam[0]["text"]) if beam else 1

        for r in range(max_rounds):
            new_beam = []
            for cand in beam:
                merged_context = {
                    "title": paper.get("title", ""),
                    "section_name": section.get("section_name"),
                    "section_text": section_text[:6000],
                    "current_summary": cand["text"],
                    "skeptic_weight": self.strategy.skeptic_weight,
                    "editor_weight": self.strategy.editor_weight,
                    "risk_weight": self.strategy.risk_weight,
                    "weaknesses": json.dumps(self._weaknesses(cand["text"], section_text), ensure_ascii=False),
                    **context,
                }
                improve_prompt = self.prompt_loader.from_file("improve_summary", self.cfg, merged_context)
                improved = self.call_llm(improve_prompt, context)  # async-safe
                s = self._score_candidate(improved, section_text)

                new_beam.append({
                    "origin": cand["origin"],
                    "variant": f"{cand['variant']}+r{r+1}",
                    "text": improved,
                    "meta": cand.get("meta", {}),
                    "score": s,
                })

            # Sort by verified then overall
            new_beam.sort(key=lambda x: (x["score"]["verified"], x["score"]["overall"]), reverse=True)

            # Diversity guard: if all candidates share the same origin, swap weakest with best other-origin from scored
            origins = {b["origin"] for b in new_beam}
            if len(origins) == 1:
                alt = next((c for c in scored if c["origin"] not in origins), None)
                if alt:
                    # replace weakest
                    new_beam[-1] = alt

            # Budget guard (marginal reward per 1k tokens)
            curr_best = new_beam[0]["score"]["overall"] if new_beam else prev_best
            curr_tokens = _est_tokens(new_beam[0]["text"]) if new_beam else prev_tokens
            marginal = _marginal(prev_best, curr_best, prev_tokens, curr_tokens)
            if marginal < min_marg:
                _logger.debug(f"[arena] early-stop@r={r+1} marginal={marginal:.3f} < {min_marg}")
                beam = new_beam[:beam_w]
                # snapshot beam before breaking
                iters.append([{"variant": b["variant"], "overall": b["score"]["overall"], "k": b["score"]["k"]} for b in beam])
                break

            # Plateau stop (no meaningful improvement)
            best_hist.append(curr_best)
            if len(best_hist) >= 2 and (best_hist[-1] - best_hist[-2]) < plateau_eps:
                _logger.debug(f"[arena] plateau-stop@r={r+1} Δ={best_hist[-1]-best_hist[-2]:.4f} < {plateau_eps}")
                beam = new_beam[:beam_w]
                iters.append([{"variant": b["variant"], "overall": b["score"]["overall"], "k": b["score"]["k"]} for b in beam])
                break

            # Commit round & continue
            beam = new_beam[:beam_w]
            iters.append([{"variant": b["variant"], "overall": b["score"]["overall"], "k": b["score"]["k"]} for b in beam])
            prev_best, prev_tokens = curr_best, curr_tokens

        # Winner selection
        winner = beam[0] if beam else (scored[0] if scored else {"text": "", "score": {}})

        self._origin_router.update(winner["origin"], float(winner["score"]["overall"]))

        self.logger.log("ArenaWinner", {
            "paper_id": str(paper.get("id") or paper.get("doc_id")),
            "section": section.get("section_name"),
            "origin": winner.get("origin"),
            "score_overall": round(winner["score"]["overall"], 3),
            "k": round(winner["score"]["k"], 3),
            "c": round(winner["score"]["c"], 3),
            "g": round(winner["score"]["g"], 3),
            "rounds": len(iters),
        })

        return {
            "winner": winner,
            "beam": beam,
            "initial_pool": scored,
            "iterations": iters,
        }

    def _persist_arena(self, case, paper, section, arena, context) -> None:
        """
        Persist arena artifacts for audit & reuse, plus compact telemetry.

        Writes:
        - arena_candidate (initial_pool, capped)
        - arena_beam      (final beam, capped)
        - arena_winner
        - arena_round_metrics (JSON: per-round summary w/o large texts)

        And attributes on the case:
        - arena_beam_width (num)
        - arena_rounds     (num)
        - arena_winner_origin (text)
        - arena_winner_overall/k/c/g (num each)
        - arena_winner_sidequest_id (text, if present)
        """
        def _meta(base=None, **kw):
            m = {
                "paper_id": str(paper.get("id") or paper.get("doc_id")),
                "section_name": section.get("section_name"),
                "type": "arena",
            }
            if base:
                m.update(base)
            m.update(kw)
            return m

        try:
            pipeline_run_id = context.get("pipeline_run_id")
            pool_cap = int(self.cfg.get("persist_pool_cap", 30))
            beam_cap = int(self.cfg.get("persist_beam_cap", 10))

            # -------- Initial pool (scored candidates) --------
            for c in (arena.get("initial_pool") or [])[:pool_cap]:
                self.memory.casebooks.add_scorable(
                    case_id=case.id,
                    role="arena_candidate",
                    pipeline_run_id=pipeline_run_id,
                    text=c.get("text", "") or "",
                    meta=_meta(
                        origin=c.get("origin"),
                        variant=c.get("variant"),
                        score=c.get("score", {}),
                        source=(c.get("meta") or {}).get("source"),
                    ),
                )

            # -------- Final beam (last round’s top-K) --------
            for b in (arena.get("beam") or [])[:beam_cap]:
                self.memory.casebooks.add_scorable(
                    case_id=case.id,
                    role="arena_beam",
                    pipeline_run_id=pipeline_run_id,
                    text=b.get("text", "") or "",
                    meta=_meta(
                        origin=b.get("origin"),
                        variant=b.get("variant"),
                        score=b.get("score", {}),
                    ),
                )

            # -------- Winner --------
            w = arena.get("winner") or {}

            # Auto-citation (uses embedding service if available)
            try:
                winner_txt = w["text"] or ""
                claims = self._extract_claim_sentences(winner_txt)
                if claims:
                    emb = self.memory.embedding
                    support_pool = (arena.get("initial_pool") or [])[:20]  # cap
                    citations = []
                    if emb:
                        cvecs = [(c, emb.get_or_create((c["text"] or "")[:2000])) for c in support_pool]
                        for s in claims:
                            svec = emb.get_or_create(s)
                            # best supporting source
                            best = None
                            best_sim = 0.0
                            for cand, v in cvecs:
                                sim = self._cos_sim(svec, v)
                                if sim > best_sim:
                                    best_sim, best = sim, cand
                            citations.append({
                                "claim": s,
                                "support_origin": best.get("origin") if best else None,
                                "support_variant": best.get("variant") if best else None,
                                "similarity": round(best_sim, 3),
                            })
                    if citations:
                        self.memory.casebooks.add_scorable(
                            case_id=case.id,
                            role="arena_citations",
                            text=dumps_safe({"citations": citations}),
                            pipeline_run_id=context.get("pipeline_run_id"),
                            meta=_meta(origin=w["origin"], variant=w["variant"])
                        )
            except Exception as e:
                _logger.warning(f"auto-citation skipped: {e}")

            w_score = w.get("score") or {}
            self.memory.casebooks.add_scorable(
                case_id=case.id,
                role="arena_winner",
                pipeline_run_id=pipeline_run_id,
                text=w.get("text", "") or "",
                meta=_meta(
                    origin=w.get("origin"),
                    variant=w.get("variant"),
                    score=w_score,
                    sidequest_id=(w.get("meta") or {}).get("sidequest_id"),
                    artifact_kind=(w.get("meta") or {}).get("artifact_kind"),
                ),
            )

            # -------- Round metrics (compact JSON) --------
            rounds_summary = []
            for ri, round_beam in enumerate(arena.get("iterations") or []):
                # each entry in iterations is already a small dict per beam item (variant/overall/k)
                rounds_summary.append({"round": ri, "beam": round_beam})

            if rounds_summary:
                self.memory.casebooks.add_scorable(
                    case_id=case.id,
                    role="arena_round_metrics",
                    pipeline_run_id=pipeline_run_id,
                    text=json.dumps({"rounds": rounds_summary}, ensure_ascii=False),
                    meta=_meta(),
                )

            # -------- Case attributes (typed; one value_* each) --------
            # numeric attrs
            try:
                self.memory.casebooks.set_case_attr(
                    case.id, "arena_beam_width", value_num=float(self.cfg.get("beam_width", 5))
                )
                self.memory.casebooks.set_case_attr(
                    case.id, "arena_rounds", value_num=float(len(arena.get("iterations") or []))
                )
                if "overall" in w_score:
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_overall", value_num=float(w_score["overall"])
                    )
                if "k" in w_score:
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_k", value_num=float(w_score["k"])
                    )
                if "c" in w_score:
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_c", value_num=float(w_score["c"])
                    )
                if "g" in w_score:
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_g", value_num=float(w_score["g"])
                    )
            except Exception:
                # do not fail the run on attribute write issues
                pass

            # text attrs
            try:
                case_id = case.id
                if w.get("origin"):
                    self.memory.casebooks.set_case_attr(
                        case_id, "arena_winner_origin", value_text=str(w["origin"])
                    )
                sid = (w.get("meta") or {}).get("sidequest_id")
                if sid:
                    self.memory.casebooks.set_case_attr(
                        case_id, "arena_winner_sidequest_id", value_text=str(sid)
                    )
                # Always set provenance attrs (were mistakenly nested under if sid)
                self.memory.casebooks.set_case_attr(
                    case_id, "arena_case_id", value_text=str(case_id)
                )
                self.memory.casebooks.set_case_attr(
                    case_id, "arena_paper_id", value_text=str(paper.get("id") or paper.get("doc_id"))
                )
                self.memory.casebooks.set_case_attr(
                    case_id, "arena_section_name", value_text=str(section.get("section_name"))
                )
                self.memory.casebooks.set_case_attr(
                    case_id, "arena_agent_name", value_text=self.name
                )

            except Exception as e:
                _logger.warning(f"Failed to set case provenance attrs: {str(e)}")

            # -------- Optional SIS card (non-blocking) --------
            try:
                sis = getattr(self.container, "get_service", lambda *_: None)("sis")
                if sis:
                    cards = [
                        {"type": "metric", "title": "Winner Overall", "value": round(float(w_score.get("overall", 0.0)), 3)},
                        {"type": "bar", "title": "K/C/G", "series": [
                            {"name": "K", "value": round(float(w_score.get("k", 0.0)), 3)},
                            {"name": "C", "value": round(float(w_score.get("c", 0.0)), 3)},
                            {"name": "G", "value": round(float(w_score.get("g", 0.0)), 3)},
                        ]},
                        {"type": "list", "title": "Winner", "items": [
                            f"Origin: {w.get('origin')}",
                            f"Variant: {w.get('variant')}",
                            f"Rounds: {len(arena.get('iterations') or [])}",
                        ]},
                    ]
                    sis.publish_cards({
                        "scope": "arena",
                        "key": f"paper:{paper.get('id') or paper.get('doc_id')}|sec:{section.get('section_name')}",
                        "title": "Arena Winner",
                        "cards": cards,
                        "meta": {
                            "case_id": case.id,
                            "paper_id": str(paper.get("id") or paper.get("doc_id")),
                            "section_name": section.get("section_name"),
                        },
                    })
            except Exception:
                pass

        except Exception as e:
            # Never let persistence explode the agent; log and continue
            try:
                self.logger.log("ArenaPersistError", {
                    "err": str(e),
                    "paper_id": str(paper.get("id") or paper.get("doc_id")),
                    "section": section.get("section_name"),
                })
            except Exception:
                _logger.warning(f"_persist_arena failed: {e}")


    # NOTE: removed accidental duplicate _persist_arena definition that overwrote the real one
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

    def _extract_claim_sentences(self, text: str) -> List[str]:
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (text or '').strip()) if len(s.strip()) > 0]
        # keep sentences that look assertive (toy heuristic)
        keep = [s for s in sents if any(w in s.lower() for w in ["show", "demonstrate", "evidence", "we", "results", "prove", "suggest"])]
        return keep[:8]  # cap

    def _cos_sim(self, a: List[float], b: List[float]) -> float:
        num = sum(x*y for x,y in zip(a,b))
        na = (sum(x*x for x in a))**0.5 or 1.0
        nb = (sum(x*x for x in b))**0.5 or 1.0
        return max(0.0, min(1.0, num/(na*nb)))

    def _rubric_dims(self, text: str, ref: str) -> Tuple[float, float]:
        import re

        sents = [s for s in re.split(r"[.!?]\s+", (text or "").strip()) if s]
        avg_len = sum(len(s.split()) for s in sents) / max(1, len(sents))
        clarity = max(0.0, min(1.0, 1.1 - (abs(avg_len - 22) / 22)))

        def toks(t):
            return set(re.findall(r"\b\w+\b", (t or "").lower()))

        inter = len(toks(text) & toks(ref))
        grounding = max(0.0, min(1.0, inter / max(30, len(toks(ref)) or 1)))
        return clarity, grounding

    def _weaknesses(self, summary: str, ref: str) -> List[str]:
        out = []
        if len(summary or "") < 400:
            out.append("too short / thin detail")
        if (
            "we propose" in (ref or "").lower()
            and "we propose" not in (summary or "").lower()
        ):
            out.append("misses core claim language")
        if (summary or "").count("(") != (summary or "").count(")"):
            out.append("formatting/parens issues")
        return out

    # ---------------- evolution / evidence ----------------

    def _evolve_strategy(self, iters: List[Dict[str, Any]], context: Optional[Dict[str, Any]]):
        if len(iters) < 2:
            # still record a point so we can analyze later
            self._record_strategy_state(context, tag="pre_change")
            return

        gains = [iters[i]["score"] - iters[i-1]["score"] for i in range(1, len(iters))]
        avg_gain = sum(gains) / len(gains) if gains else 0.0

        old_strategy = {
            "verification_threshold": self.strategy.verification_threshold,
            "skeptic_weight": self.strategy.skeptic_weight,
            "editor_weight": self.strategy.editor_weight,
            "risk_weight": self.strategy.risk_weight,
            "version": self.strategy.version,
        }

        # always record the current state (for later comparison)
        self._record_strategy_state(context, tag="pre_change")

        # propose (don’t apply yet)
        proposed = self._propose_strategy_changes(avg_gain)

        # A/B enroll for *next* work unit
        #  - A keeps current knobs
        #  - B uses proposed knobs
        if random.random() < 0.5:
            # switch to proposed for the next work unit
            self.strategy = proposed
            group = "B"
            # bump version only when actually switching
            self.strategy.version += 1
        else:
            group = "A"

        # record the assignment
        self._record_strategy_test(context, old_strategy=old_strategy, new_strategy=proposed, test_group=group, avg_gain=avg_gain)

        # keep the in-memory evolution event so your longitudinal view can see changes
        if group == "B":
            new_strategy = {
                "verification_threshold": self.strategy.verification_threshold,
                "skeptic_weight": self.strategy.skeptic_weight,
                "editor_weight": self.strategy.editor_weight,
                "risk_weight": self.strategy.risk_weight,
                "version": self.strategy.version,
            }
            event = {
                "avg_gain": round(avg_gain, 4),
                "change_amount": None,  # already in deltas above
                "old": old_strategy,
                "new": new_strategy,
                "iteration_count": len(iters),
                "timestamp": time.time(),
            }
            self._evolution_log.append(event)
            _logger.debug(f"LfL_Strategy_Evolved(AB): {event}")
            if context is not None:
                context.setdefault("strategy_evolution", []).append(event)

    def _collect_longitudinal_metrics(self) -> Dict[str, Any]:
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
                cases = self.memory.casebooks.get_cases_for_casebook(cb.id)
                for case in cases:
                    for s in self.memory.casebooks.list_scorables(case.id):
                        try:
                            payload = (
                                json.loads(s.text)
                                if isinstance(s.text, str)
                                else (s.text or {})
                            )
                            final_scores = (payload or {}).get(
                                "final_scores"
                            ) or {}
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

            vs, it = out["verification_scores"], out["iteration_counts"]
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
                out["iteration_reduction_pct"] = -1.0 * self._percent_change(
                    it[0], it[-1]
                )

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
        except Exception as e:
            try:
                self.logger.log("LfL_Longitudinal_Failed", {"err": str(e)})
            except Exception:
                pass
        return out

    def _generate_evidence_report(self, longitudinal: Dict[str, Any], cross: Dict[str, Any]) -> str:
        if not longitudinal or longitudinal.get("total_papers", 0) < 3:
            return ""
        
        score_trend = longitudinal.get("score_improvement_pct", 0.0)
        iter_trend  = longitudinal.get("iteration_reduction_pct", 0.0)
        arrow_score = "↑" if score_trend > 0 else "↓"
        arrow_iter  = "↓" if iter_trend > 0 else "↑"
        
        lines = []
        lines.append("## Learning from Learning: Evidence Report")
        lines.append("")
        lines.append(f"- **Total papers processed**: {longitudinal.get('total_papers', 0)}")
        lines.append(f"- **Verification score trend**: {score_trend:.1f}% {arrow_score}")
        lines.append(f"- **Average iterations trend**: {iter_trend:.1f}% {arrow_iter}")
        lines.append(f"- **Strategy evolution events**: {max(0, len(set(longitudinal.get('strategy_versions', []))) - 1)}")
        
        # Use cross instead of longitudinal for these metrics
        lines.append(f"- **Knowledge transfer rate**: {cross.get('knowledge_transfer_rate', 0.0):.0%}")
        ab = cross.get("strategy_ab_validation") or {}
        if ab:
            lines.append(f"- **A/B delta (B−A)**: {ab.get('delta_B_minus_A', 0.0):+.3f} (A n={ab.get('samples_A',0)}, B n={ab.get('samples_B',0)})")
        lines.append(f"- **Cross-episode evidence strength**: {cross.get('cross_episode_evidence_strength', 0.0):.0%}")
        lines.append("")
        
        vs = longitudinal.get("verification_scores", [])
        it = longitudinal.get("iteration_counts", [])
        if vs and it:
            lines.append("### Snapshot") 
            lines.append(f"- First: score={vs[0]:.2f}, iterations={it[0]}")
            lines.append(f"- Latest: score={vs[-1]:.2f}, iterations={it[-1]}")
        
        # Add cross-episode examples
        if cross and cross.get("knowledge_transfer_examples"):
            ex = cross["knowledge_transfer_examples"][0]
            lines.append("")
            lines.append("### Cross-Episode Example")
            lines.append(f"- *{ex['from_paper']}* → *{ex['to_paper']}* reused patterns:")
            for p in ex.get("patterns_used", [])[:3]:
                lines.append(f" • {p['name']} – {p['description']}")
            lines.append(f"- Mean overall after transfer: {ex.get('performance_impact', 0.0):.3f}")
        
        return "\n".join(lines)

    # ---------------- persistence ----------------
    def _save_section_to_casebook(
        self,
        casebook: CaseBookORM,
        goal_id: int,
        doc_id: str,
        section_name: str,
        section_text: str,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> CaseORM:
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
            prompt_text=dumps_safe(case_prompt),
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
            text=dumps_safe(result.get("initial_draft", {})),
            role="initial_draft",
            meta=_smeta(),
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=dumps_safe(result.get("refined_draft", {})),
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
            text=dumps_safe(result.get("verification_report", {})),
            role="verification_report",
            meta=_smeta(),
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=dumps_safe(result.get("final_validation", {})),
            role="final_validation",
            meta=_smeta(),
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=dumps_safe(
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

    def _persist_pairs(
        self,
        case_id: int,
        baseline: str,
        improved: str,
        metrics: Dict[str, Any],
        context: Dict[str, Any],
    ):
        try:
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
                    pipeline_run_id=context.get("pipeline_run_id"),
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

    # ---------------- small helpers ----------------

    def _t0(self):
        return time.time()

    def _ms_since(self, t0):
        return round((time.time() - t0) * 1000, 1)

    def _percent_change(self, start: float, end: float) -> float:
        try:
            if start is None or end is None or start == 0:
                return 0.0
            return (end - start) / abs(start) * 100.0
        except Exception:
            return 0.0

    def _estimate_tokens(self, text: str) -> int:
        # rough 4 chars/token heuristic
        return max(1, int(len(text or "") / 4))

    def _marginal_reward(self, prev_best: float, curr_best: float, prev_tokens: int, curr_tokens: int) -> float:
        delta_r = curr_best - prev_best
        delta_t = max(1, curr_tokens - prev_tokens)
        # reward per 1k toks
        return (delta_r / delta_t) * 1000.0

    def _progress_start_paper(self, paper: Dict[str, Any], sections: List[Dict[str, Any]]):
        doc_id = paper.get("id") or paper.get("doc_id")
        title = paper.get("title", "")
        self.progress.start_paper(doc_id, title, len(sections))

    def _progress_stage(self, section: Dict[str, Any], stage: str, **kv):
        self.progress.stage(stage, section.get("section_name", "unknown"), **kv)

    def _progress_start_section(self, section: Dict[str, Any], index: int):
        self.progress.start_section(section.get("section_name", "unknown"), index)

    def _progress_end_section(self, case: CaseORM, section: Dict[str, Any], metrics: Dict[str, Any]):
        # Emit event
        self.progress.end_section(section.get("section_name", "unknown"), metrics)

        # Optional: write typed case attributes for UI/queries
        if not self.progress_attrs:
            return
        try:
            ov = float(metrics.get("overall", metrics.get("scores", {}).get("overall", 0.0)))
            iters = int(metrics.get("refinement_iterations", len(metrics.get("iterations", [])) if metrics.get("iterations") else 0))
            self.memory.casebooks.set_case_attr(case.id, "progress_overall", value_num=ov)
            self.memory.casebooks.set_case_attr(case.id, "progress_iterations", value_num=float(iters))
            # mark section done
            self.memory.casebooks.set_case_attr(case.id, "progress_state", value_text="done")
        except Exception:
            pass

    def _get_paper_performance(self, casebook: CaseBookORM) -> Dict[str, float]:
        """Mean final score & iterations for a casebook."""
        scores, iters = [], []
        for case in self.memory.casebooks.get_cases_for_casebook(casebook.id) or []:
            for s in self.memory.casebooks.list_scorables(case.id) or []:
                if s.role == "metrics":
                    try:
                        rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                        final = (rec or {}).get("final_scores") or {}
                        if "overall" in final:
                            scores.append(float(final["overall"]))
                        if "refinement_iterations" in rec:
                            iters.append(int(rec["refinement_iterations"]))
                    except Exception:
                        pass
        return {
            "mean_overall": (sum(scores) / len(scores)) if scores else 0.0,
            "mean_iters": (sum(iters) / len(iters)) if iters else 0.0,
            "n": len(scores),
        }

    def _extract_verification_patterns(self, casebook: CaseBookORM) -> Dict[str, int]:
        """
        Very lightweight signal: tally winner origins per paper (e.g., 'chat_corpus', 'casebook_winner', etc.)
        """
        tally = {}
        for case in self.memory.casebooks.get_cases_for_casebook(casebook.id) or []:
            try:
                # attribute first (if you persisted it), else look at arena_winner meta
                got = False
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "arena_winner":
                        meta = (s.meta or {})
                        origin = meta.get("origin")
                        if origin:
                            tally[origin] = tally.get(origin, 0) + 1
                            got = True
                            break
                if not got:
                    # fall back to any attr you might have written
                    pass
            except Exception:
                pass
        return tally

    def _check_pattern_usage(self, curr_casebook: CaseBookORM, prev_patterns: Dict[str, int]) -> Dict[str, Any]:
        """Count how many of previous winner origins show up again."""
        used = []
        tally = self._extract_verification_patterns(curr_casebook)
        for k in prev_patterns.keys():
            if tally.get(k, 0) > 0:
                used.append({"name": k, "description": f"Winner origin reused: {k}"})
        return {"count": len(used), "patterns": used}

    def _calculate_knowledge_transfer(self) -> Dict[str, Any]:
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_action) or []
        if len(cbs) < 2:
            return {"rate": 0.0, "examples": []}
        transfer, examples = 0, []
        for i in range(1, len(cbs)):
            prev_cb, curr_cb = cbs[i-1], cbs[i]
            prev_patts = self._extract_verification_patterns(prev_cb)
            curr_perf = self._get_paper_performance(curr_cb)
            usage = self._check_pattern_usage(curr_cb, prev_patts)
            if usage["count"] > 0:
                transfer += 1
                examples.append({
                    "from_paper": getattr(prev_cb, "name", str(prev_cb.id)),
                    "to_paper": getattr(curr_cb, "name", str(curr_cb.id)),
                    "patterns_used": usage["patterns"],
                    "performance_impact": curr_perf["mean_overall"],
                })
        return {"rate": transfer / max(1, (len(cbs) - 1)), "examples": examples}

    def _calculate_domain_learning(self) -> Dict[str, Any]:
        """
        If your goal_meta injected domains into casebooks, aggregate mean_overall by tag.
        Here we approximate by using the existing casebook tag (self.casebook_action),
        which still shows adaptation across runs.
        """
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_action) or []
        buckets = {"all": []}
        for cb in cbs:
            perf = self._get_paper_performance(cb)
            buckets["all"].append(perf["mean_overall"])
        return {
            "all_mean": (sum(buckets["all"]) / len(buckets["all"])) if buckets["all"] else 0.0,
            "samples": len(buckets["all"])
        }

    def _calculate_meta_patterns(self) -> Dict[str, Any]:
        """
        Count share of sections where plateau/early-stop triggered in arena.
        We leverage 'arena_round_metrics' presence as a proxy; you could encode plateau flags there too.
        """
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_action) or []
        rounds = 0
        sections = 0
        for cb in cbs:
            for case in self.memory.casebooks.get_cases_for_casebook(cb.id) or []:
                sections += 1
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "arena_round_metrics":
                        try:
                            j = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            rounds += len((j or {}).get("rounds", []))
                        except Exception:
                            pass
        return {
            "avg_rounds": (rounds / sections) if sections else 0.0,
            "sections": sections,
        }

    def _calculate_adaptation_rate(self) -> float:
        # fraction of evolution events that actually changed knobs (B enrollments)
        if not self._evolution_log:
            return 0.0
        versions = [e["new"]["version"] for e in self._evolution_log if "new" in e]
        return len(set(versions)) / max(1, len(versions))

    def _calculate_evidence_strength(self, kt: Dict[str, Any], dom: Dict[str, Any], meta: Dict[str, Any], adapt_rate: float) -> float:
        # simple bounded blend in [0,1]
        return max(0.0, min(1.0, 0.35 * kt.get("rate", 0.0) + 0.25 * (dom.get("all_mean", 0.0)) + 0.20 * (meta.get("avg_rounds", 0.0)/5.0) + 0.20 * adapt_rate))

    def _calculate_cross_episode_metrics(self) -> Dict[str, Any]:
        """Calculate metrics that specifically prove cross-episode learning"""
        # 1. Knowledge transfer rate: how often patterns from paper N help with paper N+1
        knowledge_transfer = self._calculate_knowledge_transfer()
        
        # 2. Domain-specific learning: how verification strategies evolve for different domains
        domain_learning = self._calculate_domain_learning()
        
        # 3. Meta-pattern recognition: how the system recognizes when certain section types need different verification
        meta_patterns = self._calculate_meta_patterns()
        
        # 4. Strategy adaptation rate: how quickly the system adapts its strategy based on performance
        adaptation_rate = self._calculate_adaptation_rate()
        
        return {
            "knowledge_transfer_rate": knowledge_transfer["rate"],
            "knowledge_transfer_examples": knowledge_transfer["examples"][:3],
            "domain_learning_patterns": domain_learning,
            "meta_pattern_recognition": meta_patterns,
            "strategy_adaptation_rate": adaptation_rate,
            "cross_episode_evidence_strength": self._calculate_evidence_strength(
                knowledge_transfer, domain_learning, meta_patterns, adaptation_rate
            )
        }

    def _get_strategy_test_results(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get results of recent strategy A/B tests from scorables (not attributes)"""
        results = []
        casebooks = self.memory.casebooks.get_casebooks_by_tag(self.casebook_action) or []
        
        for cb in casebooks:
            for case in self.memory.casebooks.get_cases_for_casebook(cb.id) or []:
                group, ts, perf = None, 0.0, None
                
                # Look for A/B enrollment record
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "strategy_ab_enroll":
                        try:
                            rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            group = rec.get("test_group")
                            ts = float(rec.get("timestamp", 0.0))
                        except Exception:
                            pass
                    
                    # Look for performance metrics
                    elif s.role == "metrics":
                        try:
                            rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            final = (rec or {}).get("final_scores") or {}
                            if "overall" in final:
                                perf = float(final["overall"])
                        except Exception:
                            pass
                
                # Record valid test results
                if group in ("A", "B") and isinstance(perf, (int, float)):
                    results.append({
                        "group": group, 
                        "performance": perf, 
                        "timestamp": ts,
                        "case_id": case.id
                    })
        
        # Sort by timestamp and limit to recent results
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results[:self.cfg.get("strategy_test_history", 20)]

    def _validate_strategy_effectiveness(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate which strategy performed better and commit or revert"""
        test_results = self._get_strategy_test_results(context)
        
        if not test_results or len(test_results) < 10:  # Need enough data
            return None
        
        # Calculate performance difference between A and B groups
        perf_a = [r["performance"] for r in test_results if r["group"] == "A"]
        perf_b = [r["performance"] for r in test_results if r["group"] == "B"]
        
        if not perf_a or not perf_b:
            return None
        
        avg_perf_a = sum(perf_a) / len(perf_a)
        avg_perf_b = sum(perf_b) / len(perf_b)
        improvement = (avg_perf_b - avg_perf_a) / avg_perf_a * 100
        
        # Log the validation result
        validation_result = {
            "samples_A": len(perf_a),
            "samples_B": len(perf_b),
            "mean_A": avg_perf_a,
            "mean_B": avg_perf_b,
            "delta_B_minus_A": avg_perf_b - avg_perf_a,
            "improvement_pct": improvement,
            "timestamp": time.time()
        }
        
        self.logger.log("StrategyAB_Validation", validation_result)
        
        # Optional: automatically revert if improvement is negative
        min_improvement = self.cfg.get("min_strategy_improvement", 2.0)
        if improvement < -min_improvement:
            better_strategy = self._determine_better_strategy(test_results)
            self.strategy = better_strategy
            
            self.logger.log("StrategyReverted", {
                "reason": "negative_improvement",
                "improvement_pct": improvement,
                "reverted_to": vars(better_strategy),
                "timestamp": time.time()
            })
        
        return validation_result

    def _determine_better_strategy(self, test_results: List[Dict[str, Any]]) -> Strategy:
        """Determine which strategy performed better based on test results"""
        perf_a = [r["performance"] for r in test_results if r["group"] == "A"]
        perf_b = [r["performance"] for r in test_results if r["group"] == "B"]
        
        if not perf_a or not perf_b:
            return self.strategy  # No change if no data
        
        avg_perf_a = sum(perf_a) / len(perf_a)
        avg_perf_b = sum(perf_b) / len(perf_b)
        
        # Get the strategy data for the better performing group
        if avg_perf_b >= avg_perf_a:
            # Find the most recent B test with strategy data
            for result in test_results:
                if result["group"] == "B":
                    case_id = result["case_id"]
                    for s in self.memory.casebooks.list_scorables(case_id):
                        if s.role == "strategy_ab_enroll":
                            try:
                                rec = json.loads(s.text) if isinstance(s.text, str) else s.text
                                # _record_strategy_test stores keys 'new' and 'old', not 'new_strategy'
                                if rec.get("test_group") == "B" and "new" in rec:
                                    new_strat = rec["new"]
                                    return Strategy(
                                        verification_threshold=new_strat["verification_threshold"],
                                        skeptic_weight=new_strat["skeptic_weight"],
                                        editor_weight=new_strat["editor_weight"],
                                        risk_weight=new_strat["risk_weight"],
                                        version=new_strat["version"]
                                    )
                            except Exception:
                                pass
        else:
            # Find the most recent A test with strategy data
            for result in test_results:
                if result["group"] == "A":
                    case_id = result["case_id"]
                    for s in self.memory.casebooks.list_scorables(case_id):
                        if s.role == "strategy_ab_enroll":
                            try:
                                rec = json.loads(s.text) if isinstance(s.text, str) else s.text
                                if rec.get("test_group") == "A" and "old" in rec:
                                    old_strat = rec["old"]
                                    return Strategy(
                                        verification_threshold=old_strat["verification_threshold"],
                                        skeptic_weight=old_strat["skeptic_weight"],
                                        editor_weight=old_strat["editor_weight"],
                                        risk_weight=old_strat["risk_weight"],
                                        version=old_strat["version"]
                                    )
                            except Exception:
                                pass
        
        # Fallback to current strategy
        return self.strategy

    def _get_case_performance(self, case: CaseORM) -> float:
        """Get verification performance for a case"""
        for s in self.memory.casebooks.list_scorables(case.id):
            if s.role == "metrics":
                try:
                    metrics = json.loads(s.text) if isinstance(s.text, str) else s.text
                    return float(metrics.get("final_scores", {}).get("overall", 0.0))
                except Exception:
                    pass
        return 0.0
    
    def _save_strategy_version(self, context: Dict[str, Any]):
        """Save current strategy as a versioned artifact"""
        version_id = f"strategy_v{self.strategy.version}"
        
        # Save strategy configuration
        self.memory.models.register(
            name="learning_strategy",
            version=version_id,
            path=f"models/strategies/{version_id}.json",
            meta={
                "verification_threshold": self.strategy.verification_threshold,
                "skeptic_weight": self.strategy.skeptic_weight,
                "editor_weight": self.strategy.editor_weight,
                "risk_weight": self.strategy.risk_weight,
                "created_at": time.time(),
                "paper_count": context.get("paper_count", 0),
                "performance": context.get("avg_verification_score", 0.0)
            }
        )
        
        # Save as JSON file for reference
        strategy_data = {
            "version": version_id,
            "verification_threshold": self.strategy.verification_threshold,
            "skeptic_weight": self.strategy.skeptic_weight,
            "editor_weight": self.strategy.editor_weight,
            "risk_weight": self.strategy.risk_weight,
            "created_at": time.time(),
            "paper_count": context.get("paper_count", 0),
            "performance": context.get("avg_verification_score", 0.0)
        }
        
        os.makedirs("models/strategies", exist_ok=True)
        with open(f"models/strategies/{version_id}.json", "w") as f:
            json.dump(strategy_data, f, indent=2)

    def _revert_to_best_strategy(self, context: Dict[str, Any]):
        """Revert to the best-performing strategy version"""
        # Get all strategy versions
        versions = self.memory.models.list_versions("learning_strategy")
        if not versions:
            return
        
        # Find the best-performing version
        best_version = max(versions, key=lambda v: v.meta.get("performance", 0.0)) 
        
        # Load and apply the best strategy
        strategy_data = self.memory.models.load("learning_strategy", best_version.version)
        
        self.strategy = Strategy(
            verification_threshold=strategy_data["verification_threshold"],
            skeptic_weight=strategy_data["skeptic_weight"],
            editor_weight=strategy_data["editor_weight"],
            risk_weight=strategy_data["risk_weight"],
            version=int(best_version.version.split("_")[-1])
        )
        
        self.logger.log("StrategyReverted", {
            "to_version": best_version.version,
            "performance": strategy_data["performance"],
            "reason": "better_performance"
        })
