# stephanie/agents/learning/persistence.py
from __future__ import annotations

import asyncio  # ← NEW
import json
import logging
import time
from typing import Any, Dict, List

from stephanie.models.casebook import CaseBookORM, CaseORM
from stephanie.scoring.scorable import ScorableType
from stephanie.utils.casebook_utils import generate_casebook_name
from stephanie.utils.json_sanitize import dumps_safe
from stephanie.utils.paper_utils import (build_paper_goal_meta,
                                         build_paper_goal_text)

_logger = logging.getLogger(__name__)

AGENT_NAME = "LearningFromLearningAgent"  # stable label

class Persistence:
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.casebook_action = cfg.get("casebook_action","blog")
        self.min_section_length = int(cfg.get("min_section_length", 100))


    def _emit_report(self, *, context: Dict[str, Any], **payload):
        """
        Fire-and-forget reporting event using container.get('reporting').emit(...)
        Safe to call from sync code (no await).
        """
        try:
            reporter = self.container.get("reporting")
            
            coro = reporter.emit(context=context, **payload)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                # if no running loop (rare), just ignore to keep non-blocking
                pass
        except Exception:
            # never fail persistence due to reporting
            pass

    def prepare_casebook_goal_sections(self, paper: Dict[str, Any], context: Dict[str, Any]):
        title = paper.get("title",""); doc_id = paper.get("id") or paper.get("doc_id")
        name = generate_casebook_name(self.casebook_action, title)
        pipeline_run_id = context.get("pipeline_run_id")
        cb = self.memory.casebooks.ensure_casebook(name=name, pipeline_run_id=pipeline_run_id, description=f"LfL agent runs for paper {title}", tag=self.casebook_action)
        goal = self.memory.goals.get_or_create({
            "goal_text": build_paper_goal_text(title),
            "description": "LfL: verify & improve per section",
            "meta": build_paper_goal_meta(title, doc_id, domains=self.cfg.get("domains", [])),
        }).to_dict()
        sections = self._resolve_sections(paper)
        return cb, goal, sections

    def _resolve_sections(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id = paper.get("id") or paper.get("doc_id")
        sections = self.memory.document_sections.get_by_document(doc_id) or []
        if sections:
            return [{
                "section_name": s.section_name, "section_text": s.section_text or "",
                "section_id": s.id, "order_index": getattr(s, "order_index", None),
                "attributes": {"paper_id": str(doc_id), "section_name": s.section_name, "section_index": getattr(s,"order_index",0), "case_kind": "summary"},
            } for s in sections]
        return [{
            "section_name": "Abstract",
            "section_text": f"{paper.get('title','').strip()}\n\n{paper.get('abstract','').strip()}",
            "section_id": None, "order_index": 0,
            "attributes": {"paper_id": str(doc_id), "section_name": "Abstract", "section_index": 0, "case_kind": "summary"},
        }]

    def section_is_large_enough(self, section: Dict[str, Any]) -> bool:
        return bool((section.get("section_text") or "").strip()) and len(section.get("section_text","")) >= self.min_section_length

    def create_section_case(self, casebook: CaseBookORM, paper: Dict[str, Any], section: Dict[str, Any], goal: Dict[str, Any], context: Dict[str, Any]) -> CaseORM:
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id, goal_id=goal["id"], prompt_text=f"Section: {section['section_name']}",
            agent_name=AGENT_NAME, 
            meta={"type":"section_case"},
        )
        self.memory.casebooks.set_case_attr(case.id, "paper_id", value_text=str(paper.get("id") or paper.get("doc_id")))
        self.memory.casebooks.set_case_attr(case.id, "section_name", value_text=str(section["section_name"]))
        if section.get("section_id") is not None:
            self.memory.casebooks.set_case_attr(case.id, "section_id", value_text=str(section["section_id"]))
        if section.get("order_index") is not None:
            self.memory.casebooks.set_case_attr(case.id, "section_index", value_num=float(section.get("order_index") or 0))
        self.memory.casebooks.set_case_attr(case.id, "case_kind", value_text="summary")
        self.memory.casebooks.set_case_attr(case.id, "scorable_id", value_text=str(section.get("section_id") or ""))
        self.memory.casebooks.set_case_attr(case.id, "scorable_type", value_text=str(ScorableType.DOCUMENT_SECTION))


        return case

    def save_section(self, casebook, paper, section, verify, baseline, goal_id, context) -> CaseORM:
        threshold = float(context.get("verification_threshold", 0.85))
        return self._save_section_to_casebook(
            casebook=casebook, goal_id=goal_id,
            doc_id=str(paper.get("id") or paper.get("doc_id")),
            section_name=section["section_name"], section_text=section["section_text"],
            result={
                "initial_draft":{"title": section["section_name"], "body": baseline},
                "refined_draft":{"title": section["section_name"], "body": verify["summary"]},
                "verification_report":{"scores": verify["metrics"], "iterations": verify["iterations"]},
                "final_validation":{"scores": verify["metrics"], "passed": verify["metrics"]["overall"] >= threshold},
                "passed": verify["metrics"]["overall"] >= threshold,
                "refinement_iterations": len(verify["iterations"])
            },
            context={
                **context, "paper_title": paper.get("title",""),
                "paper_id": paper.get("id") or paper.get("doc_id"),
                "section_order_index": section.get("order_index")
            }
        )

    
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
            agent_name=AGENT_NAME,
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
        final_scores = (result.get("final_validation", {}) or {}).get("scores", {}) or {}
        metrics_payload = {
            "passed": result.get("passed", False),
            "refinement_iterations": result.get("refinement_iterations", 0),
            "final_scores": final_scores,
            "knowledge_applied_iters": final_scores.get("knowledge_applied_iters", 0),
            "knowledge_applied_lift": final_scores.get("knowledge_applied_lift", 0.0),
            "ablation": context.get("ablation"),
        }

        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=dumps_safe(metrics_payload),
            role="metrics",
            meta=_smeta(),
        )

        try:
            if context.get("ablation"):
                self.memory.casebooks.add_scorable(
                    case_id=case.id,
                    pipeline_run_id=pipeline_run_id,
                    text="true",                    # keep it tiny
                    role="ablation",
                    meta=_smeta({"reason": context.get("ablation_reason", "retrieval_masked")}),
                )
        except Exception:
            pass


        try:
            self.memory.casebooks.set_case_attr(
                case.id, "knowledge_applied_iters",
                value_num=float(metrics_payload.get("knowledge_applied_iters", 0))
            )
            self.memory.casebooks.set_case_attr(
                case.id, "knowledge_applied_lift",
                value_num=float(metrics_payload.get("knowledge_applied_lift", 0.0))
            )
        except Exception:
            pass


        return case
    
    def persist_pairs(self, case_id: int, baseline: str, verify: Dict[str, Any], context: Dict[str, Any]):
        pipeline_run_id = context.get("pipeline_run_id")
        metrics = verify["metrics"]
        improved = verify["summary"]
        try:
            self.memory.casebooks.add_scorable(
                case_id=case_id, role="knowledge_pair_positive", text=improved,
                pipeline_run_id=pipeline_run_id,
                meta={"verification_score": metrics.get("overall",0.0),
                      "knowledge_score": metrics.get("knowledge_score",0.0),
                      "strategy_version": context.get("strategy_version")},
            )
            if metrics.get("overall",0.0) >= 0.85:
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="knowledge_pair_negative", text=baseline,
                    pipeline_run_id=pipeline_run_id,
                    meta={"verification_score": max(0.0, metrics.get("overall",0.0)-0.15),
                          "knowledge_score": max(0.0, metrics.get("knowledge_score",0.0)*0.7),
                          "strategy_version": context.get("strategy_version")},
                )
        except Exception:
            pass

    def persist_arena(self, case, paper, section, arena, context):
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
                if w.get("origin"):
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_origin", value_text=str(w["origin"])
                    )
                sid = (w.get("meta") or {}).get("sidequest_id")
                if sid:
                    self.memory.casebooks.set_case_attr(
                        case.id, "arena_winner_sidequest_id", value_text=str(sid)
                    )
            except Exception:
                pass

            winner_payload = {
                "stage": "arena",                 # aligns with your CBR example
                "event": "winner",                # specific event name
                "summary": "Arena winner selected",
                "agent": "Persistence",        # stable label
                "run_id": pipeline_run_id,
                "paper_id": str(paper.get("id") or paper.get("doc_id")),
                "section_name": section.get("section_name"),
                "winner": {
                    "origin": w.get("origin"),
                    "variant": w.get("variant"),
                },
                "metrics": {
                    "overall": float(w_score.get("overall", 0.0)),
                    "k": float(w_score.get("k", 0.0)),
                    "c": float(w_score.get("c", 0.0)),
                    "g": float(w_score.get("g", 0.0)),
                },
                "beam_width": int(self.cfg.get("beam_width", 5)),
                "rounds": int(len(arena.get("iterations") or [])),
                "meta": {
                    "case_id": case.id,
                    "pipeline_run_id": pipeline_run_id,
                },
            }
            self._emit_report(context=context, **winner_payload)

            # Optional: a second, dashboard-friendly summary event
            summary_payload = {
                "stage": "arena",
                "event": "summary",
                "title": "Arena Winner",
                "agent": "Persistence",        
                "run_id": pipeline_run_id,
                "cards": [
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
                ],
                "meta": {
                    "case_id": case.id,
                    "paper_id": str(paper.get("id") or paper.get("doc_id")),
                    "section_name": section.get("section_name"),
                },
            }
            self._emit_report(context=context, **summary_payload)

            
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


    @staticmethod
    def _extract_claim_sentences(text: str) -> List[str]:
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (text or '').strip()) if s.strip()]
        keep = [s for s in sents if any(w in s.lower() for w in ["show", "demonstrate", "evidence", "we", "results", "prove", "suggest"])]
        return keep[:8]

    @staticmethod
    def _cos_sim(a: List[float], b: List[float]) -> float:
        num = sum(x*y for x,y in zip(a,b))
        na = (sum(x*x for x in a))**0.5 or 1.0
        nb = (sum(x*x for x in b))**0.5 or 1.0
        return max(0.0, min(1.0, num/(na*nb)))