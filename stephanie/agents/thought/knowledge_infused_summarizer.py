# stephanie/agents/thought/knowledge_infused_summarizer.py
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.thought.anti_hallucination import AntiHallucination
from stephanie.agents.thought.figure_grounding import FigureGrounding
from stephanie.agents.thought.paper_blog import SimplePaperBlogAgent
from stephanie.models.strategy import StrategyProfile
from stephanie.scoring.scorable import ScorableFactory, ScorableType
from stephanie.utils.hash_utils import hash_text
from stephanie.utils.json_sanitize import sanitize_for_json

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Defaults --------------------
MAX_ITERS_DEFAULT = 5
MIN_GAIN_DEFAULT = 0.015
MIN_OVERALL_DEFAULT = 0.80
TARGET_CONFIDENCE_DEFAULT = 0.95
MIN_FIGURE_SCORE_DEFAULT = 0.80
VERIFICATION_THRESHOLD_DEFAULT = 0.90
CONVERGENCE_WINDOW_DEFAULT = 2
KNOWLEDGE_GRAPH_CONF_DEFAULT = 0.70
SENTS_MIN_DEFAULT = 4
SENTS_MAX_DEFAULT = 20
CBR_CASES_DEFAULT = 3
PACS_WEIGHTS_DEFAULT = {"skeptic": 0.34, "editor": 0.33, "risk": 0.33}


log = logging.getLogger(__name__)


class KnowledgeInfusedVerifierAgent(BaseAgent):
    """
    Track C: Knowledge-Infused Verifier with true Learning-From-Learning

    Adds:
      • CBR reuse of prior wins (patches/lessons)
      • PACS multi-critic refinement with role-aware re-ranking
      • HRM epistemic judge blended into overall score
      • ZeroModel visibility (ABC quality tile + iteration strips)
      • Strategy evolution (thresholds + PACS weights persisted)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # --- config knobs
        self.max_iters = int(cfg.get("max_iters", MAX_ITERS_DEFAULT))
        self.min_gain = float(cfg.get("min_gain", MIN_GAIN_DEFAULT))
        self.min_overall = float(cfg.get("min_overall", MIN_OVERALL_DEFAULT))
        self.target_confidence = float(
            cfg.get("target_confidence", TARGET_CONFIDENCE_DEFAULT)
        )
        self.min_figure_score = float(
            cfg.get("min_figure_score", MIN_FIGURE_SCORE_DEFAULT)
        )
        self.verification_threshold = float(
            cfg.get("verification_threshold", VERIFICATION_THRESHOLD_DEFAULT)
        )
        self.convergence_window = int(
            cfg.get("convergence_window", CONVERGENCE_WINDOW_DEFAULT)
        )
        self.knowledge_graph_conf = float(
            cfg.get("knowledge_graph_conf", KNOWLEDGE_GRAPH_CONF_DEFAULT)
        )
        self.cbr_cases = int(cfg.get("cbr_cases", CBR_CASES_DEFAULT))

        # feature flags
        self.use_cbr = bool(cfg.get("use_cbr", True))
        self.use_hrm = bool(cfg.get("use_hrm", True))
        self.use_zeromodel = bool(cfg.get("use_zeromodel", True))
        self.use_descendants_metric = bool(
            cfg.get("use_descendants_metric", False)
        )
        self.hrm_weight = float(cfg.get("hrm_weight", 0.10))

        # services
        self.cbr = container.get("cbr") if self.use_cbr else None
        self.scoring = container.get(
            "scoring"
        )  # exposes HRM scorer if configured
        self.zero_model_service = (
            container.get("zeromodel") if self.use_zeromodel else None
        )
        self.kbase = container.get("kbase")  # KnowledgeBaseService

        # strategy state (persist across runs)
        self.strategy_scope = cfg.get("strategy_scope", "track_c")
        self.strategy_store = container.get(
            "strategy"
        )  # StrategyProfileService
        self.strategy = self._load_strategy_profile()

        # dependencies
        self.metrics_calculator = SimplePaperBlogAgent(
            cfg, memory, container, logger
        )
        self.anti_hallucination = AntiHallucination(logger)
        self.figure_grounding = FigureGrounding(logger)

        # sentence window (align with A/B)
        self.min_sents = int(cfg.get("min_sents", SENTS_MIN_DEFAULT))
        self.max_sents = int(cfg.get("max_sents", SENTS_MAX_DEFAULT))

        # model keys
        self.model_key_ranker = cfg.get("model_key_ranker", "ranker.sicql.v1")
        self.model_key_retriever = cfg.get(
            "model_key_retriever", "retriever.mrq.v1"
        )

        self.audit_enabled = bool(cfg.get("enable_audit_report", True))
        self.report_dir = str(cfg.get("audit_report_dir", "reports/track_c"))
        os.makedirs(self.report_dir, exist_ok=True)
        self.max_time_sec = int(
            cfg.get("max_time_sec", 120)
        )  # 2 minutes default

        self.logger.log(
            "KnowledgeInfusedVerifierInit",
            {
                "max_iters": self.max_iters,
                "verification_threshold": self.verification_threshold,
                "convergence_window": self.convergence_window,
                "cbr_cases": self.cbr_cases,
                "use_cbr": self.use_cbr,
                "use_hrm": self.use_hrm,
                "use_zeromodel": self.use_zeromodel,
                "strategy_version": self.strategy.strategy_version,
            },
        )

    # -------------------- strategy persistence --------------------
    def _load_strategy_profile(self) -> StrategyProfile:
        # Prefer service; never assume memory.meta exists
        if getattr(self, "strategy_store", None):
            return self.strategy_store.load(
                agent_name=self.name, scope=self.strategy_scope
            )
        # ephemeral fallback (won't persist across runs)
        return StrategyProfile()

    def _save_strategy_profile(self, strategy: StrategyProfile):
        if getattr(self, "strategy_store", None):
            self.strategy_store.save(
                agent_name=self.name,
                profile=strategy,
                scope=self.strategy_scope,
            )
            self.strategy = strategy
            self.verification_threshold = strategy.verification_threshold

    def _derive_domain(self, paper_data, context):
        doms = context.get("domains") or []
        if doms and isinstance(doms, list):
            d = doms[0]
            return str(
                (d.get("domain") if isinstance(d, dict) else d) or "unknown"
            )
        return "unknown"

    # -------------------- entrypoint --------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.report(
            {
                "event": "start",
                "step": "KnowledgeInfusedVerifier",
                "details": "Track C verification loop with learning",
            }
        )

        documents = context.get("documents", []) or context.get(
            self.input_key, []
        )
        chat_corpus = context.get("chat_corpus", [])
        verified_outputs: Dict[Any, Dict[str, Any]] = {}

        def _extract_summary_from_text(text: str) -> str:
            m = re.search(
                r"(?mi)^##\s*Summary\s*\n(.+?)(?=^##|\Z)", text or "", re.S
            )
            return m.group(1).strip() if m else (text or "").strip()

        for doc in documents:
            doc_id = doc.get("id") or doc.get("paper_id")
            if doc_id is None:
                self.logger.log("TrackCSkipNoDocId", {"doc": str(doc)[:200]})
                continue

            # --- Track A (baseline)
            try:
                track_a_obj = (
                    self.memory.dynamic_scorables.get_latest_by_source_pointer(
                        source="paper_summarizer",
                        source_scorable_type="document",
                        source_scorable_id=int(doc_id),
                    )
                )
            except Exception as e:
                track_a_obj = None
                self.logger.log(
                    "TrackALoadError", {"doc_id": doc_id, "error": str(e)}
                )
            if not track_a_obj:
                self.logger.log(
                    "TrackAMissing",
                    {
                        "doc_id": doc_id,
                        "hint": "Ensure Track A persisted with source_scorable_id=document_id and type=document",
                    },
                )
                continue
            a_meta = self._safe_meta(track_a_obj)
            a_metrics = a_meta.get("metrics") or {}

            # --- Track B (sharpened)
            try:
                track_b_obj = (
                    self.memory.dynamic_scorables.get_latest_by_source_pointer(
                        source="sharpened_paper_summarizer",
                        source_scorable_type="dynamic",
                        source_scorable_id=int(track_a_obj.id),
                    )
                )
            except Exception as e:
                track_b_obj = None
                self.logger.log(
                    "TrackBLoadError", {"doc_id": doc_id, "error": str(e)}
                )
            if not track_b_obj:
                self.logger.log(
                    "TrackBMissing",
                    {
                        "doc_id": doc_id,
                        "hint": "Ensure Track B persisted with source_scorable_id=<Track A dynamic id> and type=dynamic",
                    },
                )
                continue
            b_meta = self._safe_meta(track_b_obj)

            b_text = (getattr(track_b_obj, "text", "") or "").strip()
            baseline_b_summary = _extract_summary_from_text(b_text) or (
                b_meta.get("summary") or b_text
            )

            title = doc.get("title", "") or (a_meta.get("title") or "")
            abstract = (
                a_meta.get("abstract")
                or b_meta.get("abstract")
                or self._fetch_abstract(doc_id)
            )
            arxiv_summary = (
                a_meta.get("arxiv_summary")
                or b_meta.get("arxiv_summary")
                or (doc.get("summary", "") or "")
            )

            baseline_b_metrics = b_meta.get("metrics")
            if not baseline_b_metrics:
                baseline_b_metrics = self._score_summary(
                    baseline_b_summary,
                    abstract,
                    arxiv_summary,
                    {},
                    title,
                    context,
                )

            # --- Track C (verify + learn)
            try:
                verified = await self._verify_summary(
                    doc_id=str(doc_id),
                    enhanced_summary=baseline_b_summary,
                    paper_data=doc,
                    chat_corpus=chat_corpus,
                    context=context,
                    track_a=track_a_obj,
                    track_b=track_b_obj,
                )
            except Exception as e:
                self.logger.log(
                    "TrackCVerifyError",
                    {
                        "doc_id": doc_id,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                continue

            verified_outputs[doc_id] = verified

            # --- training events + VPM tiles
            try:
                v_metrics = verified.get("metrics") or {}
                if v_metrics.get(
                    "overall", 0.0
                ) >= self.min_overall and verified.get(
                    "passes_guardrails", False
                ):
                    self._emit_training_events(
                        paper={
                            "paper_id": doc.get("paper_id", doc_id),
                            "title": title,
                            "abstract": abstract,
                            "author_summary": arxiv_summary,
                        },
                        baseline_summary=baseline_b_summary,
                        verified_summary=verified.get("summary", ""),
                        baseline_metrics=baseline_b_metrics,
                        verified_metrics=v_metrics,
                        context=context,
                    )

                self._emit_vpm_tiles(
                    doc_id=doc_id,
                    title=title,
                    metrics_a=a_metrics,
                    metrics_b=baseline_b_metrics or {},
                    metrics_c=v_metrics,
                    iterations_c=verified.get("iterations", []),
                    out_dir="reports/vpm",
                    lineage_ids=[
                        getattr(track_a_obj, "id", None),
                        getattr(track_b_obj, "id", None),
                    ],
                )
            except Exception as e:
                try:
                    self.memory.session.rollback()
                except Exception:
                    pass
                self.logger.log(
                    "TrackCPostProcessError",
                    {"doc_id": doc_id, "error": str(e)},
                )

        context.setdefault("summary_v2", {})
        context["summary_v2"] = verified_outputs

        if self.audit_enabled:
            context.setdefault("reports", [])
            # push all generated .md files for this batch
            # (we already called self.report() for each, but some UIs read context["reports"])
            try:
                md_files = [
                    f for f in os.listdir(self.report_dir) if f.endswith(".md")
                ]
                for f in md_files:
                    context["reports"].append(
                        {
                            "agent": self.name,
                            "type": "markdown",
                            "path": os.path.join(self.report_dir, f),
                        }
                    )
            except Exception:
                pass

        return context

    # -------------------- core verification loop --------------------
    async def _verify_summary(
        self,
        doc_id: str,
        enhanced_summary: str,
        paper_data: Dict[str, Any],
        chat_corpus: List[Dict[str, Any]],
        context: Dict[str, Any],
        track_a: Any,
        track_b: Any,
    ) -> Dict[str, Any]:
        start_time = time.time()
        time_limit_at = start_time + float(self.max_time_sec)
        abstract = self._fetch_abstract(doc_id)
        arxiv_summary = paper_data.get("summary", "")
        goal_title = paper_data.get("title", "")

        knowledge_graph = context.get("knowledge_graph")
        if not knowledge_graph:
            knowledge_graph = await self._build_knowledge_graph(
                doc_id, paper_data, chat_corpus, context
            )

        current_summary = enhanced_summary
        current_metrics = self._score_summary(
            current_summary,
            abstract,
            arxiv_summary,
            knowledge_graph,
            goal_title,
            context,
        )
        start_overall = current_metrics.get("overall", 0.0)
        best_summary, best_metrics = current_summary, current_metrics

        iterations: List[Dict[str, Any]] = []
        no_improve_count = 0
        convergence_track: List[float] = []
        lineage_ids = [
            getattr(track_a, "id", None),
            getattr(track_b, "id", None),
        ]

        audit = {
            "doc_id": str(doc_id),
            "title": goal_title,
            "start_overall": float(current_metrics.get("overall", 0.0)),
            "baseline_metrics": current_metrics,
            "iterations": [],  # we’ll append per-iter snapshots here
            "strategy_before": self.strategy.to_dict(),
            "track_a_id": getattr(track_a, "id", None),
            "track_b_id": getattr(track_b, "id", None),
            "kbase_hints": [],
        }

        for iter_idx in range(self.max_iters):
            iter_start = time.time()

            # CBR pack
            case_pack = self._retrieve_case_pack(goal_title, k=self.cbr_cases)
            self.report({"event": "cbr_pack", "k": len(case_pack)})

            # prompt with CBR
            prompt = self._build_verification_prompt(
                current_summary=current_summary,
                claims=(knowledge_graph or {}).get("claims", []),
                paper_data=paper_data,
                case_pack=case_pack,
                context=context,
                abstract=abstract,
            )
            # hash + excerpt for report (avoid dumping huge prompts verbatim)
            prompt_hash = hash_text(prompt)[:12]
            prompt_excerpt = prompt[:600]

            # PACS refinement (get details for the report)
            raw_llm = self.call_llm(prompt, context=context) or current_summary
            candidate, panel_detail = self._pacs_refine(
                raw_llm,
                abstract,
                context,
                paper_data,
                knowledge_graph,
                return_panel=True,
            )

            # score candidate
            cand_metrics = self._score_summary(
                candidate,
                abstract,
                arxiv_summary,
                knowledge_graph,
                goal_title,
                context,
            )
            gain = cand_metrics["overall"] - current_metrics["overall"]

            # emit iteration tile
            try:
                self.zero_model_service.emit_iteration_tile(
                    doc_id=str(doc_id),
                    iteration=iter_idx + 1,
                    metrics={
                        "overall": cand_metrics.get("overall", 0.0),
                        "knowledge_verification": cand_metrics.get(
                            "knowledge_verification", 0.0
                        ),
                        "hrm_score": cand_metrics.get("hrm_score", 0.0)
                        if cand_metrics.get("hrm_score") is not None
                        else 0.0,
                    },
                    output_dir="reports/vpm/iters",
                )
            except Exception as e:
                self.logger.log(
                    "VPMIterTileWarn", {"doc_id": doc_id, "error": str(e)}
                )

            # record iteration
            iter_payload = {
                "iteration": iter_idx + 1,
                "current_score": current_metrics["overall"],
                "best_candidate_score": cand_metrics["overall"],
                "gain": gain,
                "processing_time": time.time() - iter_start,
                "knowledge_graph_conf": self.knowledge_graph_conf,
                "prompt_hash": prompt_hash,
                "prompt_excerpt": prompt_excerpt,
                "panel_detail": panel_detail or {},
            }
            if knowledge_graph:
                iter_payload["claim_coverage"] = knowledge_graph.get(
                    "claim_coverage", 0.0
                )
                iter_payload["evidence_strength"] = knowledge_graph.get(
                    "evidence_strength", 0.0
                )
            iterations.append(iter_payload)
            if self.audit_enabled:
                audit["iterations"].append(iter_payload)
                if panel_detail and not audit.get("kbase_hints"):
                    audit["kbase_hints"] = panel_detail.get("kb_hints", [])

            if time.time() > time_limit_at:
                self.report(
                    {
                        "event": "early_stop",
                        "reason": "time_limit",
                        "iteration": iter_idx + 1,
                    }
                )
                break

            # accept if improves enough
            if (
                cand_metrics["overall"] >= self.min_overall
                and gain >= self.min_gain
            ):
                current_summary = candidate
                current_metrics = cand_metrics
                if cand_metrics["overall"] > best_metrics["overall"]:
                    best_summary, best_metrics = (
                        current_summary,
                        current_metrics,
                    )
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            convergence_track.append(best_metrics["overall"])

            # stops
            if best_metrics["overall"] >= self.target_confidence:
                self.report(
                    {
                        "event": "verification_converged",
                        "reason": "target_confidence",
                    }
                )
                break
            if no_improve_count >= 2:
                self.report(
                    {"event": "verification_converged", "reason": "no_improve"}
                )
                break
            if len(convergence_track) >= self.convergence_window:
                recent = convergence_track[-self.convergence_window :]
                if (
                    np.std(recent) < 5e-3
                    and (max(recent) - min(recent)) < 2e-2
                ):
                    self.report(
                        {
                            "event": "verification_converged",
                            "reason": "convergence_window",
                        }
                    )
                    break

        # guardrails
        is_valid, hallucination_issues = self._verify_hallucinations(
            best_summary, abstract, arxiv_summary, knowledge_graph
        )
        figure_results = self._verify_figure_grounding(
            best_summary, paper_data, knowledge_graph
        )

        # strategy evolution
        if best_metrics["overall"] > start_overall + self.min_gain:
            new_weights = self._adjust_pacs_weights(
                {**best_metrics, "figure_results": figure_results}
            )
            new_threshold = min(
                0.99, self.strategy.verification_threshold + 0.01
            )
            self.strategy.update(
                pacs_weights=new_weights, verification_threshold=new_threshold
            )
            self._save_strategy_profile(self.strategy)
            self.report(
                {
                    "event": "strategy_updated",
                    "new_weights": new_weights,
                    "new_threshold": new_threshold,
                }
            )

        result = {
            "summary": best_summary,
            "metrics": best_metrics,
            "iterations": iterations,
            "processing_time": time.time() - start_time,
            "hallucination_issues": hallucination_issues,
            "figure_results": figure_results,
            "passes_guardrails": bool(is_valid)
            and (
                figure_results.get("overall_figure_score", 0.0)
                >= self.min_figure_score
            ),
            "converged": best_metrics["overall"] >= self.target_confidence,
            "knowledge_graph": knowledge_graph,
            "verification_trace": {
                "iterations": len(iterations),
                "final_score": best_metrics["overall"],
                "converged": len(convergence_track) >= self.convergence_window
                and np.std(convergence_track[-self.convergence_window :])
                < 1e-2,
            },
        }

        try:
            # 1) ensure blog casebook
            paper_id = str(paper_data.get("paper_id", doc_id))
            post_slug = paper_data.get("post_slug") or "main"

            title = paper_data.get("title")
            name = f"blog::{title}"
            meta = {
                "paper_id": paper_id,
                "arxiv_id": paper_data.get("arxiv_id"),
                "title": paper_data.get("title", ""),
                "post_slug": post_slug,
            }
            casebook = self.memory.casebooks.ensure_casebook(
                name=name,
                pipeline_run_id=context.get("pipeline_run_id"),
                tags=["blog"],
                meta=meta,
            )

            meta = {}
            response_texts = [raw_llm, candidate]
            case = self.memory.casebooks.add_case(
                casebook_id=casebook.id,
                goal_id=casebook.goal_id,
                prompt_text=prompt,
                agent_name=self.name,
                response_texts=response_texts,
                meta=meta,
            )

            # 3) auto-promote champion if better than any existing champion by overall score
            # try:
            #     existing = blog.get_champion_text(casebook_name=cb.name, section="summary")
            #     should_promote = False
            #     if not existing:
            #         should_promote = True
            #     else:
            #         # crude comparison on overall; if you want, fetch champion metrics from cases for precision
            #         should_promote = float(best_metrics.get("overall", 0.0)) >= self.min_overall

            #     if should_promote:
            #         blog.mark_champion(
            #             casebook_name=casebook_name,
            #             case_id=str(add_res["case_id"]),
            #             section="summary",
            #         )
            # except Exception as e:
            #     self.logger.log("BlogChampionEvalWarn", {"error": str(e)})
        except Exception as e:
            self.logger.log(
                "BlogCasebookWriteWarn", {"doc_id": doc_id, "error": str(e)}
            )

        # persist as dynamic scorable
        try:
            safe_meta = sanitize_for_json(
                {
                    "paper_id": paper_data.get("paper_id", doc_id),
                    "title": paper_data.get("title", ""),
                    "metrics": best_metrics,
                    "origin": "track_c_verified",
                    "verification_trace": result["verification_trace"],
                    "hallucination_issues": result.get(
                        "hallucination_issues", []
                    ),
                    "origin_ids": lineage_ids,
                    "figure_results": result.get("figure_results", {}),
                }
            )

            scorable_id = self.memory.dynamic_scorables.add(
                pipeline_run_id=context.get("pipeline_run_id"),
                scorable_type=ScorableType.DYNAMIC,
                source=self.name,
                text=best_summary,
                meta=safe_meta,  # ← sanitized!
                source_scorable_id=getattr(track_b, "id", None),
                source_scorable_type="dynamic",
            )
            result["scorable_id"] = scorable_id
            scorable = self.memory.casebooks.add_scorable(
                case_id=case.id,
                pipeline_run_id=context.get("pipeline_run_id"),
                role="text",
                scorable_id=scorable_id,
                text=enhanced_summary,
                scorable_type=ScorableType.DYNAMIC,
                meta=meta,
            )
            result["case_scorable_id"] = scorable_id
        except Exception as e:
            self.logger.log(
                "DynamicScorableSaveError",
                {
                    "doc_id": doc_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

        domain = self._derive_domain(paper_data, context)
        strategy_delta = {
            "skeptic": self.strategy.pacs_weights.get("skeptic", 0.0)
            - PACS_WEIGHTS_DEFAULT["skeptic"],
            "editor": self.strategy.pacs_weights.get("editor", 0.0)
            - PACS_WEIGHTS_DEFAULT["editor"],
            "risk": self.strategy.pacs_weights.get("risk", 0.0)
            - PACS_WEIGHTS_DEFAULT["risk"],
            "verification_threshold": self.strategy.verification_threshold
            - VERIFICATION_THRESHOLD_DEFAULT,
        }
        try:
            if self.kbase:
                self.kbase.update_from_paper(
                    domain=domain,
                    summary_text=result["summary"],
                    metrics=result["metrics"],
                    iterations=result["iterations"],
                    strategy_delta=strategy_delta,
                )

            self._capture_cross_paper_signals(
                paper_id=str(paper_data.get("paper_id", doc_id)),
                domain=domain,
                metrics=result["metrics"],
                iterations=result["iterations"],
                strategy=self.strategy,
                strategy_delta=strategy_delta,
            )
        except Exception as e:
            self.logger.log(
                "KBUpdateWarn", {"doc_id": doc_id, "error": str(e)}
            )

        if self.audit_enabled:
            try:
                audit["strategy_after"] = self.strategy.to_dict()
                audit["final_metrics"] = result["metrics"]
                audit["passes_guardrails"] = result["passes_guardrails"]
                audit["hallucination_issues"] = result.get(
                    "hallucination_issues", []
                )
                audit["figure_results"] = result.get("figure_results", {})
                timeline_png = self._plot_iteration_timeline(
                    audit["iterations"],
                    out_path=os.path.join(
                        self.report_dir, f"{doc_id}_timeline.png"
                    ),
                )
                transfer_png = self.generate_transfer_curve(
                    output_path=os.path.join(
                        self.report_dir, "transfer_curve.png"
                    )
                )
                report_md = self._write_audit_report(
                    doc_id=str(doc_id),
                    title=goal_title,
                    audit=audit,
                    timeline_path=timeline_png,
                    transfer_curve_path=transfer_png,
                    abc_gif_path=result.get(
                        "quality_tile_path"
                    ),  # if ZeroModel returned one
                )
                # expose to pipeline context + report stream
                self.report(
                    {
                        "event": "verification_report",
                        "doc_id": str(doc_id),
                        "path": report_md,
                    }
                )
            except Exception as e:
                self.logger.log(
                    "AuditReportError", {"doc_id": doc_id, "error": str(e)}
                )

        return result

    # --- Cross-paper signals & evaluation ---------------------------------
    def _capture_cross_paper_signals(
        self,
        *,
        paper_id: str,
        domain: str,
        metrics: Dict[str, Any],
        iterations: List[Dict[str, Any]],
        strategy: StrategyProfile,
        strategy_delta: Dict[str, float],
    ) -> None:
        """
        Persist tiny signals that let us measure transfer across papers.
        Plays nice if tables aren't present (no hard deps).
        """
        payload = {
            "paper_id": paper_id,
            "domain": domain,
            "strategy_version": int(getattr(strategy, "strategy_version", 0)),
            "verification_threshold": float(
                getattr(strategy, "verification_threshold", 0.0)
            ),
            "pacs_weights": dict(getattr(strategy, "pacs_weights", {})),
            "strategy_delta": dict(strategy_delta or {}),
            "final_quality": float(metrics.get("overall", 0.0)),
            "knowledge_verification": float(
                metrics.get("knowledge_verification", 0.0)
            ),
            "hrm_score": float(metrics.get("hrm_score", 0.0))
            if metrics.get("hrm_score") is not None
            else None,
            "iterations": int(len(iterations or [])),
            "first_iter_score": float(
                (iterations or [{}])[0].get("current_score", 0.0)
            )
            if iterations
            else None,
            "last_iter_score": float(
                (iterations or [{}])[-1].get("best_candidate_score", 0.0)
            )
            if iterations
            else None,
            "ts": time.time(),
        }
        # Optional: calibration events (soft dependency)
        try:
            self.memory.calibration_events.add(
                {
                    "domain": domain or "general",
                    "query": f"{paper_id}:{domain}",  # any non-null string
                    "raw_similarity": float(metrics.get("overall", 0.0)),
                    "is_relevant": bool(
                        float(metrics.get("overall", 0.0)) >= self.min_overall
                    ),
                    "scorable_id": str(paper_id),
                    "scorable_type": "paper",
                    "entity_type": "summary_verification",
                    "features": {
                        "quality": float(metrics.get("overall", 0.0)),
                        "knowledge_verification": float(
                            metrics.get("knowledge_verification", 0.0)
                        ),
                        "hrm_score": None
                        if metrics.get("hrm_score") is None
                        else float(metrics["hrm_score"]),
                        "verification_threshold": float(
                            strategy.verification_threshold
                        ),
                        "pacs_weights": dict(strategy.pacs_weights or {}),
                        "iterations": int(len(iterations or [])),
                        "first_iter_score": payload.get("first_iter_score"),
                        "last_iter_score": payload.get("last_iter_score"),
                    },
                }
            )
        except Exception as e:
            log.error("CalibrationAddWarn", {"error": str(e)})

        # Optional: casebook of signals (simple append-only log)
        try:
            casebooks = getattr(self.memory, "casebooks", None)
            if casebooks and hasattr(casebooks, "add"):
                casebooks.add(
                    casebook_name="verification_signals",
                    case_id=f"{paper_id}",
                    role="signal",
                    text=json.dumps(payload),
                    meta={"domain": domain, "timestamp": payload["ts"]},
                )
        except Exception as e:
            self.logger.log("CasebookAddWarn", {"error": str(e)})

        self.logger.log(
            "CrossPaperSignalCaptured",
            {
                "paper_id": paper_id,
                "domain": domain,
                "quality": payload["final_quality"],
                "strategy_version": payload["strategy_version"],
            },
        )

    def analyze_transfer_effect(
        self, learning_split: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Reads 'verification_signals' casebook and checks if baseline performance
        (papers after the split, treated as 'no-learning' runs) rises over time.
        """
        try:
            casebooks = getattr(self.memory, "casebooks", None)
            if not (casebooks and hasattr(casebooks, "get_by_casebook")):
                self.logger.log(
                    "TransferAnalyzeSkip", {"reason": "casebooks missing"}
                )
                return None

            rows = (
                casebooks.get_by_casebook(
                    casebook_name="verification_signals", role="signal"
                )
                or []
            )
            if len(rows) < 20:
                return None

            # Expect sequential ids or anything we can sort on 'timestamp'
            data = []
            for r in rows:
                try:
                    d = json.loads(getattr(r, "text", "{}") or "{}")
                    data.append(d)
                except Exception:
                    continue
            if not data:
                return None

            data.sort(key=lambda d: d.get("ts", 0.0))
            post = [x for i, x in enumerate(data) if i >= learning_split]
            if len(post) < 10:
                return None

            # simple start/end window means
            head = post[: max(5, min(10, len(post) // 4))]
            tail = post[-max(5, min(10, len(post) // 4)) :]

            initial = (
                float(np.mean([h.get("final_quality", 0.0) for h in head]))
                if head
                else 0.0
            )
            final = (
                float(np.mean([t.get("final_quality", 0.0) for t in tail]))
                if tail
                else 0.0
            )
            improvement = final - initial

            return {
                "initial_baseline": initial,
                "final_baseline": final,
                "improvement": improvement,
                "sample_size": len(post),
                "significant": improvement > 0.05,  # coarse heuristic
            }
        except Exception as e:
            self.logger.log("TransferAnalyzeError", {"error": str(e)})
            return None

    def generate_transfer_curve(
        self, output_path: str = "reports/vpm/transfer_curve.png"
    ) -> Optional[str]:
        """
        Produce a simple PNG of baseline performance drift after the learning split.
        """
        try:
            pass
        except Exception:
            self.logger.log(
                "TransferCurveSkip", {"reason": "matplotlib not available"}
            )
            return None

        res = self.analyze_transfer_effect()
        if not res:
            return None

        # Rebuild the time series from signals
        rows = (
            self.memory.casebooks.get_by_casebook(
                casebook_name="verification_signals", role="signal"
            )
            or []
        )
        rows = sorted(rows, key=lambda r: json.loads(r.text).get("ts", 0.0))
        perf = [json.loads(r.text).get("final_quality", 0.0) for r in rows]

        plt.figure(figsize=(9, 5.2))
        plt.plot(range(1, len(perf) + 1), perf, linewidth=2)
        plt.axhline(
            y=res["initial_baseline"], linestyle="--", label="Initial baseline"
        )
        plt.axhline(
            y=res["final_baseline"], linestyle="--", label="Final baseline"
        )
        plt.title("Transfer Learning Effect: Baseline Performance Over Time")
        plt.xlabel("Paper Index")
        plt.ylabel("Quality (overall)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.log("TransferCurveSaved", {"path": output_path, **res})
        return output_path

    # -------------------- CBR / PACS / HRM helpers --------------------
    def _retrieve_case_pack(
        self, title: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        if not self.use_cbr or not self.cbr:
            return []
        try:
            cases = self.cbr.retrieve(goal_text=title, top_k=k)
            pack = []
            for c in cases or []:
                pack.append(
                    {
                        "title": (c.get("goal_text") or "")[:160],
                        "why_it_won": (
                            c.get("scores", {}).get("winner_rationale") or ""
                        )[:240],
                        "patch": (c.get("lessons") or "")[:240],
                        "summary": (
                            c.get("best_text") or c.get("summary") or ""
                        )[:400],
                    }
                )
            return pack
        except Exception as e:
            self.logger.log("CBRRetrieveError", {"error": str(e)})
            return []

    # change signature to accept paper_data and knowledge_tree
    def _pacs_refine(
        self,
        candidate: str,
        abstract: str,
        context: Dict[str, Any],
        paper_data: Dict[str, Any] | None = None,
        knowledge_tree: Dict[str, Any] | None = None,
        *,
        return_panel: bool = False,
    ) -> str:
        title = (
            (paper_data or {}).get("title")
            or context.get("goal", {}).get("goal_text", "")
            or ""
        )
        domain = self._derive_domain(paper_data or {}, context or {})
        kb_ctx = (
            self.kbase.context_for_paper(
                title=title, abstract=abstract, domain=domain
            )
            if self.kbase
            else {}
        )
        nudges = kb_ctx.get("weight_nudges", {}) or {}

        # Ephemeral weights: do NOT mutate self.strategy here
        base_w = dict(self.strategy.pacs_weights)
        work_w = dict(base_w)
        for k, dv in nudges.items():
            work_w[k] = max(0.2, min(0.4, work_w.get(k, 0.33) + float(dv)))
        s = sum(work_w.values()) or 1.0
        work_w = {k: v / s for k, v in work_w.items()}

        roles = [
            ("skeptic", "remove speculation; flag ungrounded claims"),
            (
                "editor",
                f"tighten structure; keep {self.min_sents}-{self.max_sents} sentences",
            ),
            ("risk", "require figure/table citation for any numeric claim"),
        ]

        panel: List[Tuple[str, str]] = []
        for role, brief in roles:
            prompt = f"""Role: {role.title()}. Brief: {brief}
    Abstract:
    \"\"\"{abstract[:1000]}\"\"\"

    Text to review:
    \"\"\"{candidate}\"\"\"\n
    Return ONLY the revised paragraph."""
            try:
                out = self.call_llm(prompt, context=context)
                if out:
                    panel.append((role, out.strip()))
            except Exception:
                continue
        if not panel:
            return (candidate, None) if return_panel else candidate

        # score by role...
        best_text, best_score = candidate, -1.0
        role_scores = []
        for role, text in panel:
            m = self.metrics_calculator._compute_metrics(text, abstract, "")
            if role == "risk":
                m["figure_results"] = self._verify_figure_grounding(
                    text, paper_data or {}, knowledge_tree or {}
                )
            role_score = self._role_weighted_score(role, m, weights=work_w)
            role_scores.append(
                {"role": role, "score": role_score, "metrics": m, "text": text}
            )
            if role_score > best_score:
                best_text, best_score = text, role_score

        details = {
            "kb_hints": kb_ctx.get("hints", []),
            "kb_templates_count": len(kb_ctx.get("templates", [])),
            "nudges": nudges,
            "weights_used": work_w,
            "panel": role_scores,
        }
        self.logger.log("PACSRefine", details)
        return (best_text, details) if return_panel else best_text

    def _role_weighted_score(
        self,
        role: str,
        m: Dict[str, float],
        weights: Dict[str, float] | None = None,
    ) -> float:
        skeptic_focus = 0.6 * (
            1.0 - float(m.get("hallucination_rate", 0.0))
        ) + 0.4 * float(m.get("faithfulness", 0.0))
        editor_focus = 0.5 * float(m.get("coherence", 0.0)) + 0.5 * float(
            m.get("structure", 0.0)
        )
        risk_focus = (
            float(m.get("figure_results", {}).get("overall_figure_score", 0.0))
            if isinstance(m.get("figure_results"), dict)
            else 0.0
        )
        base = float(m.get("overall", 0.0))
        wmap = weights or self.strategy.pacs_weights
        w = wmap.get(role, 0.33)

        if role == "skeptic":
            role_focus = skeptic_focus
        elif role == "editor":
            role_focus = editor_focus
        else:
            role_focus = risk_focus

        score = 0.5 * base + 0.5 * role_focus
        return w * score

    def _hrm_epistemic(
        self, text: str, goal: str, context: Dict[str, Any]
    ) -> Tuple[Optional[float], str]:
        if not self.use_hrm or not self.scoring:
            return None, ""
        try:
            scorable = ScorableFactory.from_dict(
                {"text": text, "goal": goal, "type": "document"}
            )
            bundle = self.scoring.score(
                "hrm",
                context=context,
                scorable=scorable,
                dimensions=["alignment"],
            )
            res = getattr(bundle, "results", {}).get("alignment")
            if res is None:
                return None, ""
            score = (
                float(getattr(res, "score", None))
                if getattr(res, "score", None) is not None
                else None
            )
            rationale = getattr(res, "rationale", "")
            return score, rationale
        except Exception as e:
            self.logger.log("HRMScoreError", {"error": str(e)})
            return None, ""

    def _build_verification_prompt(
        self,
        current_summary: str,
        claims: List[Dict[str, Any]],
        paper_data: Dict[str, Any],
        case_pack: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        abstract: Optional[str] = None,  # <-- add param
    ) -> str:
        title = paper_data.get("title", "")
        domain = self._derive_domain(paper_data, context or {})
        abs_text = (
            abstract
            if abstract is not None
            else self._fetch_abstract(
                paper_data.get("id") or paper_data.get("paper_id")
            )
        )
        kb_ctx = (
            self.kbase.context_for_paper(
                title=title, abstract=abs_text, domain=domain
            )
            if self.kbase
            else {}
        )
        tmpl_text = ""
        if kb_ctx.get("templates"):
            bullets = []
            for t in kb_ctx["templates"]:
                bullets.append("- " + " ".join(t.get("outline", [])[:3]))
            tmpl_text = "\n\nTemplates that worked before:\n" + "\n".join(
                bullets
            )

        hints_text = ""
        if kb_ctx.get("hints"):
            hints_text = "\n\nStrategy hints:\n" + "\n".join(
                f"- {h}" for h in kb_ctx["hints"]
            )

        claims_text = "\n".join(
            f"- {c.get('text', '').strip()}"
            for c in (claims or [])[:5]
            if c.get("text")
        )
        examples = ""
        if case_pack:
            ex_lines = []
            for ex in case_pack[:3]:
                ex_lines.append(
                    f"- Lesson: {ex.get('patch', '')}\n  Why it won: {ex.get('why_it_won', '')}"
                )
            if ex_lines:
                examples = "\n\nPrior improvements to emulate:\n" + "\n".join(
                    ex_lines
                )
        return f"""
You are a verification expert checking this academic paper summary against the paper's key claims.

Title: {title}

Key Claims:
{claims_text}{examples}{tmpl_text}{hints_text}

Current summary:
\"\"\"{current_summary}\"\"\"

Improve the summary by:
1) Ensuring all key claims are accurately represented
2) Citing figures/tables for quantitative claims when warranted
3) Removing unsupported statements 
4) Preserving clarity and neutrality

Constraints:
- Keep to {self.min_sents}-{self.max_sents} sentences
- Use ONLY facts present in the paper and allowed context
- Do not invent numbers or facts

Verified summary:
""".strip()

    def _verify_against_knowledge_tree(
        self, summary: str, knowledge_tree: Dict[str, Any]
    ) -> float:
        if not knowledge_tree:
            return 0.5
        claims = knowledge_tree.get("claims", []) or []
        covered = 0
        for claim in claims:
            text = claim.get("text", "")
            if text and self.metrics_calculator._contains_concept(
                summary, text
            ):
                covered += 1
        claim_coverage = covered / max(1, len(claims))
        rels = knowledge_tree.get("relationships", []) or []
        if self.strategy:
            threshold = self.strategy.verification_threshold
            log.debug(f"Using strategy threshold: {threshold}")
        else:
            threshold = self.verification_threshold
        strong = [
            r for r in rels if float(r.get("confidence", 0.0)) >= threshold
        ]
        evidence_strength = len(strong) / max(1, len(rels))
        return (0.7 * claim_coverage) + (0.3 * evidence_strength)

    # -------------------- guardrails --------------------
    def _verify_hallucinations(
        self,
        summary: str,
        abstract: str,
        author_summary: str,
        knowledge_tree: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        # Make AntiHallucination resilient to key-type mismatches in figure maps etc.
        try:
            is_valid, issues = self.anti_hallucination.verify_section(
                summary,
                knowledge_tree,
                {"abstract": abstract, "summary": author_summary},
            )
            return (bool(is_valid), issues or [])
        except Exception as e:
            self.logger.log("AntiHallucinationError", {"error": str(e)})
            return True, ["anti_hallucination_failed_soft"]

    def _verify_figure_grounding(
        self,
        summary: str,
        paper_data: Dict[str, Any],
        knowledge_tree: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Simple heuristic extractor for quant claims → expected to be replaced by FigureGrounding
        quant_claims = []
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", summary or "")
            if s.strip()
        ]
        for sent in sentences:
            matches = re.findall(
                r"(\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b)",
                sent,
                flags=re.I,
            )
            if matches:
                quant_claims.append(
                    {
                        "claim": sent,
                        "value": matches[0][0],
                        "metric": matches[0][1],
                        "has_citation": any(
                            marker in sent.lower()
                            for marker in [
                                "figure",
                                "fig.",
                                "table",
                                "as shown",
                                "see",
                            ]
                        ),
                    }
                )
        properly_cited = sum(1 for c in quant_claims if c.get("has_citation"))
        citation_rate = properly_cited / max(1, len(quant_claims))
        return {
            "total_claims": len(quant_claims),
            "properly_cited": properly_cited,
            "citation_rate": citation_rate,
            "overall_figure_score": citation_rate,
            "claims": quant_claims,
        }

    # -------------------- strategy evolution --------------------
    def _adjust_pacs_weights(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        weights = dict(self.strategy.pacs_weights)
        if float(metrics.get("knowledge_verification", 0.0)) > 0.8:
            weights["editor"] = min(0.4, weights.get("editor", 0.33) + 0.05)
            weights["skeptic"] = max(0.2, weights.get("skeptic", 0.33) - 0.05)
        if float(metrics.get("hallucination_rate", 1.0)) > 0.2:
            weights["skeptic"] = min(0.4, weights.get("skeptic", 0.33) + 0.05)
            weights["editor"] = max(0.2, weights.get("editor", 0.33) - 0.05)
        if float(metrics.get("coverage", 0.0)) < 0.7:
            weights["skeptic"] = min(0.4, weights.get("skeptic", 0.33) + 0.03)
        if float(metrics.get("coherence", 0.0)) < 0.7:
            weights["editor"] = min(0.4, weights.get("editor", 0.33) + 0.03)
        fig_score = 0.0
        fr = metrics.get("figure_results")
        if isinstance(fr, dict):
            fig_score = float(fr.get("overall_figure_score", 0.0))
        if fig_score < 0.7:
            weights["risk"] = min(0.4, weights.get("risk", 0.33) + 0.03)
        # normalize
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    # -------------------- utilities --------------------
    def _safe_meta(self, obj) -> dict:
        meta = getattr(obj, "meta", {}) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return meta

    def _emit_vpm_tiles(
        self,
        *,
        doc_id,
        title: str,
        metrics_a: dict | None,
        metrics_b: dict | None,
        metrics_c: dict | None,
        iterations_c: list[dict] | None,
        out_dir: str = "reports/vpm",
        lineage_ids: List[Any] | None = None,
    ):
        try:
            svc = self.zero_model_service
            if not svc:
                self.logger.log(
                    "VPMSkipServiceMissing",
                    {"doc_id": doc_id, "reason": "zero_model service missing"},
                )
                return
            vpm_data = self._prepare_vpm_data(
                doc_id,
                title,
                metrics_a or {},
                metrics_b or {},
                metrics_c or {},
                iterations_c or [],
            )
            if hasattr(svc, "generate_summary_vpm_tiles"):
                result = svc.generate_summary_vpm_tiles(
                    vpm_data=vpm_data, output_dir=out_dir
                )
            else:
                # minimal fallback: ABC triptych only
                names = [
                    "overall",
                    "coverage",
                    "faithfulness",
                    "structure",
                    "no_halluc",
                ]
                import numpy as _np

                rows = []
                for label, mm in (
                    ("A", metrics_a),
                    ("B", metrics_b),
                    ("C", metrics_c),
                ):
                    mm = mm or {}
                    rows.append(
                        [
                            float(mm.get("overall", 0.0)),
                            float(mm.get("claim_coverage", 0.0)),
                            float(mm.get("faithfulness", 0.0)),
                            float(mm.get("structure", 0.0)),
                            float(1.0 - mm.get("hallucination_rate", 1.0)),
                        ]
                    )
                mat = _np.asarray(rows, dtype=_np.float32)
                out = f"{out_dir}/{doc_id}_abc.gif"
                if hasattr(svc, "_emit_timeline"):
                    svc._emit_timeline(mat, out)
                result = {"quality_tile_path": out}
            self.logger.log(
                "VPMTilesGenerated", {"doc_id": doc_id, **(result or {})}
            )
        except Exception as e:
            self.logger.log(
                "VPMTileGenerationError",
                {
                    "doc_id": doc_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

    def _prepare_vpm_data(
        self, doc_id, title, metrics_a, metrics_b, metrics_c, iterations_c
    ):
        def pack(m):
            # Map into a compact, consistent bundle
            return {
                "overall": float(m.get("overall", 0.0)),
                "coverage": float(
                    m.get("claim_coverage", m.get("coverage", 0.0))
                ),
                "faithfulness": float(m.get("faithfulness", 0.0)),
                "structure": float(m.get("structure", 0.0)),
                "no_halluc": float(1.0 - m.get("hallucination_rate", 1.0)),
                "figure_ground": float(
                    (m.get("figure_results", {}) or {}).get(
                        "overall_figure_score", 0.0
                    )
                )
                if isinstance(m.get("figure_results"), dict)
                else 0.0,
            }

        return {
            "doc_id": doc_id,
            "title": title[:80],
            "metrics": {
                "A": pack(metrics_a),
                "B": pack(metrics_b),
                "C": pack(metrics_c),
            },
            "iterations": iterations_c or [],
            "timestamp": time.time(),
        }

    def _fetch_abstract(self, doc_id) -> str:
        try:
            sections = self.memory.document_sections.get_by_document(doc_id)
            for s in sections:
                sd = s.to_dict()
                if (
                    sd.get("section_name") or ""
                ).lower().strip() == "abstract":
                    return sd.get("section_text", "") or ""
        except Exception as e:
            self.logger.log(
                "AbstractFetchFailed", {"doc_id": doc_id, "error": str(e)}
            )
        return ""

    def _emit_training_events(
        self,
        paper: Dict[str, Any],
        baseline_summary: str,
        verified_summary: str,
        baseline_metrics: Dict[str, float],
        verified_metrics: Dict[str, float],
        context: Dict[str, Any],
    ):
        title = paper.get("title", "paper")
        gain = float(
            verified_metrics.get("overall", 0.0)
            - (baseline_metrics or {}).get("overall", 0.0)
        )
        w = max(0.1, min(1.0, gain + 0.3))

        # pointwise
        self.memory.training_events.add_pointwise(
            model_key=self.model_key_retriever,
            dimension="alignment",
            query_text=title,
            cand_text=verified_summary,
            label=1,
            weight=float(verified_metrics.get("overall", 0.7)),
            trust=float(verified_metrics.get("overall", 0.7)),
            goal_id=context.get("goal", {}).get("id"),
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            source="track_c",
            meta={
                "stage": "track_c",
                "gain": gain,
                "knowledge_verification": verified_metrics.get(
                    "knowledge_verification", 0.0
                ),
            },
        )

        # pairwise vs. Track B
        self.memory.training_events.insert_pairwise(
            model_key=self.model_key_ranker,
            dimension="alignment",
            query_text=title,
            pos_text=verified_summary,
            neg_text=baseline_summary,
            weight=w,
            trust=w * 0.6,
            goal_id=context.get("goal", {}).get("id"),
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            source="track_c",
            meta={
                "stage": "track_c",
                "verified_score": verified_metrics.get("overall"),
                "baseline_score": (baseline_metrics or {}).get("overall"),
                "gain": gain,
            },
        )

        # optional pairwise vs author/arXiv summary
        author_summary = paper.get("author_summary", "") or ""
        if author_summary.strip():
            author_metrics = self._score_summary(
                author_summary,
                paper.get("abstract", ""),
                author_summary,
                {},
                title,
                context,
            )
            prefer_verified = verified_metrics.get(
                "overall", 0.0
            ) > author_metrics.get("overall", 0.0)
            pos = verified_summary if prefer_verified else author_summary
            neg = author_summary if prefer_verified else verified_summary
            self.memory.training_events.insert_pairwise(
                model_key=self.model_key_ranker,
                dimension="alignment",
                query_text=title,
                pos_text=pos,
                neg_text=neg,
                weight=0.5,
                trust=0.3,
                goal_id=context.get("goal", {}).get("id"),
                pipeline_run_id=context.get("pipeline_run_id"),
                agent_name=self.name,
                source="track_c",
                meta={
                    "stage": "track_c",
                    "verified_score": verified_metrics.get("overall", 0.0),
                    "author_score": author_metrics.get("overall", 0.0),
                    "prefer_verified": prefer_verified,
                },
            )

    async def _build_knowledge_graph(
        self,
        doc_id: str,
        paper_data: Dict[str, Any],
        chat_corpus: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Minimal KG builder that ONLY uses the service's build_tree(...).
        Returns a normalized dict with expected keys.
        """

        def _empty_kg() -> Dict[str, Any]:
            return {
                "nodes": [],
                "relationships": [],
                "claims": [],
                "claim_coverage": 0.0,
                "evidence_strength": 0.0,
                "temporal_coherence": 0.0,
                "domain_alignment": 0.0,
                "knowledge_gaps": [],
                "meta": {"paper_id": str(doc_id)},
            }

        def _normalize(kg: Any) -> Dict[str, Any]:
            if not isinstance(kg, dict):
                kg = {}
            kg = kg.get("knowledge_graph") or kg
            if not isinstance(kg, dict):
                kg = {}
            # ensure expected fields
            kg.setdefault("nodes", [])
            kg.setdefault("relationships", [])
            kg.setdefault("claims", [])
            kg.setdefault("claim_coverage", 0.0)
            kg.setdefault("evidence_strength", 0.0)
            kg.setdefault("temporal_coherence", 0.0)
            kg.setdefault("domain_alignment", 0.0)
            kg.setdefault("knowledge_gaps", [])
            kg.setdefault("meta", {})
            kg["meta"].setdefault("paper_id", str(doc_id))
            return kg

        svc = self.container.get("knowledge_graph")
        if not (svc and hasattr(svc, "build_tree")):
            self.logger.log("KGMissingBuildTree", {"doc_id": doc_id})
            return _empty_kg()

        paper_text = (paper_data.get("text") or "").strip()
        try:
            # build_tree is sync; run it in a worker so we don't block the event loop
            kg = await asyncio.to_thread(
                svc.build_tree,
                paper_text=paper_text,
                paper_id=str(doc_id),
                chat_corpus=chat_corpus or [],
                trajectories=context.get("conversation_trajectories", [])
                or [],
                domains=context.get("domains", []) or [],
            )
            self.logger.log(
                "KGBuildPath",
                {"service": svc.__class__.__name__, "method": "build_tree"},
            )
            return _normalize(kg)
        except Exception as e:
            self.logger.log(
                "KnowledgeGraphBuildFailed",
                {
                    "doc_id": doc_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            return _empty_kg()

    # --- Emit one VPM tile per iteration (safe wrapper) ---
    def _emit_vpm_tile(
        self,
        doc_id: str,
        stage: str,
        metrics: Dict[str, Any],
        lineage_ids: List[Any] | None,
        context: Dict[str, Any],
    ) -> None:
        """
        Best-effort tile emission. If a ZeroModel/visualization service is
        available, we forward the request; otherwise we no-op.
        """
        try:
            zm = getattr(
                self, "zero_model_service", None
            ) or self.container.get("zeromodel")
        except Exception:
            zm = None

        if not zm:
            # Nothing to do; keep pipeline robust
            self.logger.log(
                "VPMTileSkipNoService", {"doc_id": doc_id, "stage": stage}
            )
            return

        try:
            payload = {
                "doc_id": str(doc_id),
                "stage": stage,
                "metrics": dict(metrics or {}),
                "lineage": [x for x in (lineage_ids or []) if x is not None],
                "vpf": {
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    "agent": self.name,
                    "stage": stage,
                },
            }
            # Delegate to the service; tolerate either method name
            fn = (
                getattr(zm, "create_tile", None)
                or getattr(zm, "generate_summary_vpm_tiles", None)
                or getattr(zm, "emit_tile", None)
            )
            if callable(fn):
                fn(**payload) if fn.__code__.co_kwonlyargcount else fn(payload)
            else:
                self.logger.log(
                    "VPMTileSkipNoAPI", {"doc_id": doc_id, "stage": stage}
                )
        except Exception as e:
            self.logger.log(
                "VPMTileEmitError",
                {
                    "doc_id": doc_id,
                    "stage": stage,
                    "error": str(e),
                },
            )

    # --- Scoring (with knowledge + optional HRM) ---
    def _score_summary(
        self,
        summary: str,
        abstract: str,
        author_summary: str,
        knowledge_tree: Dict[str, Any],
        goal_title: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Base Track-A metrics + knowledge verification (+ optional HRM).
        Compatible with older callers that don’t pass goal_title/context.
        """
        # 1) deterministic base metrics
        base = self.metrics_calculator._compute_metrics(
            summary, abstract, author_summary
        )

        # 2) knowledge verification (claim coverage + evidence strength)
        ver = self._verify_against_knowledge_tree(summary, knowledge_tree)

        # 3) optional HRM epistemic judge
        hrm_score = None
        try:
            if self.use_hrm:
                _goal = goal_title or (
                    context.get("goal", {}).get("goal_text", "")
                )
                hrm_score, _ = self._hrm_epistemic(summary, _goal, context)
        except Exception:
            hrm_score = None

        # normalize HRM to [0,1]
        if hrm_score is not None:
            hs = float(hrm_score)
            if hs < 0.0 or hs > 1.0:
                # treat as a logit-like raw signal
                hs = 1.0 / (1.0 + math.exp(-hs))
                self.logger.log(
                    "HRMScoreNormalized", {"raw": hrm_score, "norm": hs}
                )
            # hard clamp
            hrm_score = max(0.0, min(1.0, hs))

        # 4) blend
        #   keep your prior weighting; add a small HRM term if present
        overall = 0.8 * base.get("overall", 0.0) + 0.2 * ver
        if hrm_score is not None:
            overall = (
                1.0 - self.hrm_weight
            ) * overall + self.hrm_weight * float(hrm_score)

        out = dict(base)
        out["knowledge_verification"] = float(ver)
        if hrm_score is not None:
            out["hrm_score"] = float(hrm_score)
        out["overall"] = float(overall)
        return out

    def _plot_iteration_timeline(
        self, iters: List[Dict[str, Any]], out_path: str
    ) -> Optional[str]:
        if not iters:
            return None

        xs = [it["iteration"] for it in iters]
        ys = [float(it.get("best_candidate_score", 0.0)) for it in iters]
        cs = [float(it.get("current_score", 0.0)) for it in iters]

        plt.figure(figsize=(8.6, 4.2))
        plt.plot(xs, cs, linewidth=2, label="current score")
        plt.plot(xs, ys, linewidth=2, label="candidate score")
        plt.title("Track C: Per-Iteration Verification Scores")
        plt.xlabel("Iteration")
        plt.ylabel("Overall score")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path

    def _write_audit_report(
        self,
        *,
        doc_id: str,
        title: str,
        audit: Dict[str, Any],
        timeline_path: Optional[str],
        transfer_curve_path: Optional[str],
        abc_gif_path: Optional[str] = None,
    ) -> str:
        """
        Render a compact Markdown report that shows:
        - overview & baseline vs. final metrics,
        - iteration timeline figure,
        - PACS panel snapshots (scores per role),
        - knowledge verification, hallucination & figure grounding,
        - strategy shifts,
        - transfer learning curve (global).
        """

        def f(x):  # short float
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)

        base = audit.get("baseline_metrics", {})
        final = audit.get("final_metrics", {})
        issues = audit.get("hallucination_issues", [])
        figure = audit.get("figure_results", {})
        hints = audit.get("kbase_hints", [])
        strat_b = audit.get("strategy_before", {})
        strat_a = audit.get("strategy_after", {})

        lines = []
        lines.append(f"# Verification Report — {title or doc_id}")
        lines.append("")
        lines.append(
            f"**Doc ID:** `{doc_id}`  |  **Start overall:** {f(audit.get('start_overall'))}  |  **Final overall:** {f(final.get('overall', 0.0))}"
        )
        lines.append("")
        if abc_gif_path:
            lines.append(
                f"![ABC tile]({os.path.relpath(abc_gif_path, self.report_dir)})"
            )
            lines.append("")

        # Overview table
        lines.append("## Overview (Baseline → Final)")
        lines.append("")
        rows = [
            ("overall", base.get("overall"), final.get("overall")),
            (
                "knowledge_verification",
                base.get("knowledge_verification"),
                final.get("knowledge_verification"),
            ),
            (
                "coverage",
                base.get("claim_coverage", base.get("coverage")),
                final.get("claim_coverage", final.get("coverage")),
            ),
            (
                "faithfulness",
                base.get("faithfulness"),
                final.get("faithfulness"),
            ),
            ("structure", base.get("structure"), final.get("structure")),
            (
                "hallucination_rate (↓)",
                base.get("hallucination_rate"),
                final.get("hallucination_rate"),
            ),
            (
                "figure_grounding",
                (base.get("figure_results") or {}).get("overall_figure_score")
                if isinstance(base.get("figure_results"), dict)
                else None,
                (final.get("figure_results") or {}).get("overall_figure_score")
                if isinstance(final.get("figure_results"), dict)
                else None,
            ),
        ]
        lines.append("| metric | baseline | final |")
        lines.append("|---|---:|---:|")
        for k, b, c in rows:
            lines.append(f"| {k} | {f(b)} | {f(c)} |")
        lines.append("")

        # Iteration timeline
        if timeline_path:
            rel = os.path.relpath(timeline_path, self.report_dir)
            lines.append("## Iteration Timeline")
            lines.append("")
            lines.append(f"![Iteration scores]({rel})")
            lines.append("")

        # Iteration snapshots (compact)
        lines.append("## Iteration Snapshots")
        lines.append("")
        for it in audit.get("iterations", []):
            lines.append(
                f"### Iter {it['iteration']} — gain: {f(it.get('gain', 0.0))}, cand: {f(it.get('best_candidate_score'))}"
            )
            lines.append(f"- prompt: `{it.get('prompt_hash')}` — excerpt:")
            excerpt = (it.get("prompt_excerpt") or "").replace("\n", " ")
            lines.append(
                f"  > {excerpt[:240]}{'…' if len(excerpt) > 240 else ''}"
            )
            pd = it.get("panel_detail") or {}
            if pd.get("weights_used"):
                w = pd["weights_used"]
                lines.append(
                    f"- PACS weights used: skeptic={f(w.get('skeptic'))}, editor={f(w.get('editor'))}, risk={f(w.get('risk'))}"
                )
            # show top-1 panel improvement
            best = None
            for entry in pd.get("panel") or []:
                if not best or float(entry.get("score", -1)) > float(
                    best.get("score", -1)
                ):
                    best = entry
            if best:
                lines.append(
                    f"- Best panel: **{best['role']}** (score {f(best['score'])})"
                )
            lines.append("")

        # Knowledge verification & guardrails
        lines.append("## Knowledge Verification & Guardrails")
        lines.append("")
        lines.append(
            f"- Claim coverage (final): {f(final.get('claim_coverage', final.get('coverage')))}"
        )
        lines.append(
            f"- Evidence strength (final): {f(audit['iterations'][-1].get('evidence_strength') if audit.get('iterations') else None)}"
        )
        if issues:
            lines.append(
                f"- Hallucination issues: {len(issues)} (listed below)"
            )
        if isinstance(figure, dict):
            lines.append(
                f"- Figure grounding: {figure.get('properly_cited', 0)}/{figure.get('total_claims', 0)} cited (rate={f(figure.get('citation_rate'))})"
            )
        lines.append("")
        if issues:
            lines.append("<details><summary>Hallucination issues</summary>\n")
            for x in issues[:20]:
                lines.append(f"- {str(x)[:240]}")
            lines.append("\n</details>\n")

        # Strategy evolution
        lines.append("## Strategy Evolution")
        lines.append("")
        try:
            wb = (
                (strat_b.get("pacs_weights") or {})
                if isinstance(strat_b, dict)
                else {}
            )
            wa = (
                (strat_a.get("pacs_weights") or {})
                if isinstance(strat_a, dict)
                else {}
            )
            lines.append(
                f"- Threshold: {f((strat_b or {}).get('verification_threshold'))} → {f((strat_a or {}).get('verification_threshold'))}"
            )
            lines.append(
                f"- PACS weights: skeptic {f(wb.get('skeptic'))}→{f(wa.get('skeptic'))}, editor {f(wb.get('editor'))}→{f(wa.get('editor'))}, risk {f(wa.get('risk'))}→{f(wa.get('risk'))}"
            )
        except Exception:
            pass
        if hints:
            lines.append("")
            lines.append("### KBase Hints Applied")
            for h in hints:
                lines.append(f"- {h}")
            lines.append("")

        # Transfer (global)
        if transfer_curve_path:
            rel = os.path.relpath(transfer_curve_path, self.report_dir)
            lines.append("## Transfer Learning Trend (Global)")
            lines.append("")
            lines.append(f"![Transfer curve]({rel}) But how you doing")
            lines.append("")

        # finalize
        out_md = os.path.join(self.report_dir, f"{doc_id}.md")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return out_md
