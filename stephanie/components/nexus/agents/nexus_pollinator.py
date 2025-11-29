# stephanie/components/nexus/agents/nexus_pollinator.py
from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics as stats
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.nexus.blossom.runner import BlossomRunnerAgent
from stephanie.components.nexus.blossom.viewer.renderer import \
    write_garden_frames
from stephanie.components.nexus.graph.builder import GraphBuilder
from stephanie.components.nexus.graph.graph import (GraphScore, NexusGraph,
                                                    NexusGraphConfig,
                                                    NexusViewManifest)
from stephanie.components.nexus.graph.knowledge_index import \
    compute_knowledge_index_from_files
from stephanie.scoring.metrics.scorable_processor import ScorableProcessor
from stephanie.scoring.scorable import Scorable
from stephanie.services.bus.events.prompt_job import Priority
from stephanie.services.bus.prompt_client import PromptClient
from stephanie.services.bus.prompt_worker import PromptDispatcherWorker
from stephanie.services.scoring_service import ScoringService
from stephanie.utils.json_sanitize import dumps_safe
from stephanie.utils.progress_mixin import ProgressMixin

log = logging.getLogger(__name__)

# --------------------------- CONFIG ---------------------------


@dataclass(slots=True)
class NexusPollinatorConfig:
    k_candidates: int = 2
    sharpen_iters: int = 1
    novelty_tau: float = 0.12
    promote_margin: float = 0.02
    dims: List[str] = field(
        default_factory=lambda: [
            "alignment",
            "faithfulness",
            "coverage",
            "clarity",
            "coherence",
        ]
    )
    scorers: List[str] = field(default_factory=lambda: ["sicql", "mrq", "hrm"])
    scorer_weights: Dict[str, float] = field(default_factory=dict)
    dimension_weights: Dict[str, float] = field(default_factory=dict)
    use_llm_heuristic: bool = False
    use_vpm_phi: bool = False
    persist_scores: bool = True
    blossom_cfg: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)

    offload_mode: str = "await"  # "await" | "fire_and_forget" | "disabled"
    max_inflight: int = 64  # global inflight LLM jobs
    response_timeout_s: float = 90.0
    response_poll_ms: int = 250

    attr_enrichment: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "min_columns": 120,          # treat as “already enriched” if >= this many columns
        "baseline_topq": 0.9,        # enrich parents in top 10% after baseline
        "cands_topk": 0,             # 0 = skip; else enrich top-K candidates by fused score
        "winner_always": True,       # always enrich the chosen winner
        "max_total": 200,            # safety budget per run
        "max_inflight": 4,           # parallelism cap
        "ttl_minutes": 1440          # if you add staleness checks later
    })

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> NexusPollinatorConfig:
        bc = dict(cfg.get("blossom", {}))
        scoring = dict(cfg.get("scoring", {}))
        return NexusPollinatorConfig(
            k_candidates=int(cfg.get("k_candidates", 2)),
            sharpen_iters=int(cfg.get("sharpen_iters", 1)),
            novelty_tau=float(cfg.get("novelty_tau", 0.12)),
            promote_margin=float(cfg.get("promote_margin", 0.02)),
            dims=list(
                cfg.get(
                    "dims",
                    [
                        "alignment",
                        "faithfulness",
                        "coverage",
                        "clarity",
                        "coherence",
                    ],
                )
            ),
            scorers=list(cfg.get("scorers", ["sicql", "mrq", "hrm"])),
            scorer_weights=dict(cfg.get("scorer_weights", {})),
            dimension_weights=dict(cfg.get("dimension_weights", {})),
            use_llm_heuristic=bool(cfg.get("use_llm_heuristic", False)),
            use_vpm_phi=bool(cfg.get("use_vpm_phi", False)),
            persist_scores=bool(cfg.get("persist_scores", True)),
            blossom_cfg=bc,
            scoring=scoring,
            offload_mode=str(
                cfg.get("offload_mode", bc.get("offload_mode", "await"))
            ),
            max_inflight=int(
                cfg.get("max_inflight", bc.get("max_inflight", 64))
            ),
            response_timeout_s=float(
                cfg.get(
                    "response_timeout_s", bc.get("response_timeout_s", 90.0)
                )
            ),
            response_poll_ms=int(
                cfg.get("response_poll_ms", bc.get("response_poll_ms", 250))
            ),
            attr_enrichment=dict(cfg.get("attr_enrichment", {})) or {
                "enabled": True, "min_columns": 120, "baseline_topq": 0.9,
                "cands_topk": 0, "winner_always": True, "max_total": 200,
                "max_inflight": 4, "ttl_minutes": 1440
            },
        )


# --------------------------- AGENT ---------------------------


class NexusPollinatorAgent(ProgressMixin, BaseAgent):
    """
    Improves a cohort of Scorables via Blossom (generate -> refine -> select),
    evaluates lift, persists training events, and reports progress in real time.

    Progress tasks:
      - overall:  "nexus_improver:{pipeline_run_id|epoch}"
      - per-ep:   "blossom_episode:parent={parent_id}"
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self._init_progress(container, logger)
        self.cfg = NexusPollinatorConfig.from_cfg(cfg or {})
        self.scoring: ScoringService = container.get("scoring")
        self.blossom = BlossomRunnerAgent(
            self.cfg.blossom_cfg, memory, container, logger
        )
        self.scorable_processor = ScorableProcessor(
            cfg=cfg, memory=memory, container=container, logger=logger
        )
        self.graph_builder = GraphBuilder(
            cfg=self.cfg, memory=self.memory, container=self.container, logger=self.logger
        )
        self._attr_budget = int(self.cfg.attr_enrichment.get("max_total", 200))
        self._attr_inflight = asyncio.Semaphore(int(self.cfg.attr_enrichment.get("max_inflight", 4)))
        self._enriched_ids: set[str] = set()  

    # --------------------------- PUBLIC ---------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cache = self.container.get("cache")
        if cache:
            await self.memory.ensure_bus_connected()
            cache.set_bus(self.memory.bus)     # critical line    
            self.logger.log("CacheServiceInitialized", {"type": type(cache).__name__})

        prompt_worker = PromptDispatcherWorker(
            cfg=self.cfg,
            memory=self.memory,
            container=self.container,
            logger=self.logger,
        )
        await prompt_worker.start()
        self.cfg.scorer_weights = self._normalize_weights(
            self.cfg.scorer_weights
        )
        self.cfg.dimension_weights = self._normalize_weights(
            self.cfg.dimension_weights
        )

        goal_text = (context.get("goal") or {}).get("goal_text", "") or ""
        items: List[Scorable] = self._to_scorables(
            context.get("scorables", [])
        )
        if not items:
            context[self.output_key] = {"status": "no_items"}
            return context
        parent_rows = await self._materialize_rows(items, context)

        run_id = context.get("pipeline_run_id")
        run_dir = self._run_dir(context)

        # NEW: optional Nexus graph handle for this run
        nexus_graph = self._get_nexus_graph(context)

        # ---------- (A) BASELINE GRAPH (GraphBuilder) ----------
        baseline_manifest = self._manifest_from_scorables(run_id, items)
        baseline_path = run_dir / "baseline_graph.json"
        g_report = self.graph_builder.build(
            run_id,
            baseline_manifest,
            namespace="vpm:baseline",
            knn_k=8,
            sim_threshold=0.35,
            add_temporal=True,
            add_mst_backbone=True,
            add_domain_edges=True,
            add_entity_edges=True,
            export_path=str(baseline_path),
        )
        context["nexus_graph_json"] = str(baseline_path)

        old_baseline_path = run_dir / "old_graph_baseline.json"
        self._write_graph(g_report, old_baseline_path)
        context["nexus_graph_json_old"] = str(
            old_baseline_path
        )  # viewer needs this

        # collect per-episode telemetry
        context.setdefault("blossom_runs", [])

        # sync runner defaults
        try:
            self.blossom.run_cfg.return_top_k = int(self.cfg.k_candidates)
            self.blossom.run_cfg.sharpen_top_k = int(
                self.cfg.k_candidates if self.cfg.sharpen_iters > 0 else 0
            )
        except Exception:
            pass

        use_vpm_phi = bool(context.get("use_vpm_phi", self.cfg.use_vpm_phi))

        # ----- PROGRESS: overall task -----
        task_run = self._task_run_name(context)
        self.pstart(task_run, total=len(items))

        # ---------------- 1) Baseline eval ----------------
        self.pstage(task_run, stage="baseline")

        base_evals: Dict[str, float] = {}
        base_vectors: Dict[str, Dict[str, float]] = {}
        all_vectors: Dict[str, Dict[str, float]] = {}  # NEW: global dim cache
        final_cohort: List[Scorable] = []  # <- we'll fill with winners/parents
        improved_qualities: Dict[str, float] = {}
        improved_vectors: Dict[str, Dict[str, float]] = {}

        for idx, s in enumerate(items, start=1):
            parent_scorable = s
            eval_res = self._eval_state(
                parent_scorable, context, use_vpm_phi=use_vpm_phi
            )
            sid = self._sid(parent_scorable, idx)
            dims_vec = dict(eval_res.get("dims") or {})
            base_evals[sid] = float(eval_res["overall"])
            base_vectors[sid] = dims_vec
            all_vectors[sid] = dims_vec  # NEW
            if self.cfg.persist_scores:
                self._persist_dims(parent_scorable, eval_res, context)

        baseline_summary = self._cohort_summary(list(base_evals.values()))

        # --- JIT enrichment for top-quantile parents (post-baseline) ---
        try:
            q = float(self.cfg.attr_enrichment.get("baseline_topq", 0.9))
        except Exception as _e:
            self._slog("AttrEnrichBaselineTopQError", {"error": str(_e)})

        if nexus_graph is not None:
            try:
                nexus_graph.ingest_manifest(
                    baseline_manifest,
                    qualities=base_evals,
                    metrics=all_vectors,
                    phase="baseline",
                )
                if isinstance(g_report, dict) and g_report.get("graph_json"):
                    val = g_report["graph_json"]
                    if isinstance(val, str):
                        val = json.loads(Path(val).read_text())
                    nexus_graph.write_edges_from_graph_json(
                        val,
                        phase="baseline",
                        channel="vpm:baseline",
                    )
            except Exception as e:
                self._slog(
                    "NexusBaselineSyncError",
                    {"error": str(e)},
                )

        # ---------------- 2) Blossom + local select --------
        self.pstage(task_run, stage="blossom")

        decisions: List[Dict[str, Any]] = []
        total = len(items)
        run_dir = self._run_dir(context)

        for idx, s in enumerate(items, start=1):
            parent_scorable = s

            # ---- emit garden event ----
            pid = getattr(
                parent_scorable, "blossom_node_id", None
            ) or self._sid(parent_scorable, idx)
            self._emit_garden_event(
                run_dir,
                "episode_start",
                parent_id=str(pid),
                parent_text_len=len(parent_scorable.text or ""),
                parent_overall=base_evals[self._sid(parent_scorable, idx)],
            )

            # ---- per-episode subtask ----
            ep_task = self._task_episode_name(
                getattr(parent_scorable, "id", "unknown")
            )
            est_total = max(
                1, self.cfg.k_candidates * max(1, self.cfg.sharpen_iters)
            )
            self.pstart(ep_task, total=est_total)

            use_offload = bool(
                self.cfg.blossom_cfg.get("offload_prompts", False)
            )
            mode = self.cfg.offload_mode

            if use_offload and mode in ("await", "fire_and_forget"):
                # 1) Enqueue K prompts quickly
                tickets = await self._offload_candidate_prompts(
                    parent_scorable,
                    {**context, "progress_task": ep_task},
                    k=self.cfg.k_candidates,
                    model=self.cfg.blossom_cfg.get("model", "gpt-4o-mini"),
                    target_pool=self.cfg.blossom_cfg.get(
                        "target_pool", "provider"
                    ),
                )

                if mode == "fire_and_forget":
                    # mark pending and move on; a collector will finalize later
                    context.setdefault("pending_prompt_tickets", []).extend(
                        tickets
                    )
                    decisions.append(
                        {
                            "parent_id": getattr(parent_scorable, "id", None),
                            "status": "pending_prompts",
                            "k_requested": self.cfg.k_candidates,
                        }
                    )
                    self.pstep(task_run, n=1)
                    self.pdone(ep_task)
                    continue

                # mode == "await"  →  2) Wait concurrently (bounded globally)
                texts = await self._await_offloaded_candidates(
                    tickets,
                    timeout_s=self.cfg.response_timeout_s,
                    poll_ms=self.cfg.response_poll_ms,
                )

                # 3) Convert results -> children Scorables
                children: List[Scorable] = []
                for i, txt in enumerate(texts):
                    if not txt:
                        continue
                    children.append(
                        Scorable(
                            id=f"bl_off:{getattr(parent_scorable, 'id', 'x')}:{i}",
                            text=str(txt),
                            target_type=getattr(
                                parent_scorable, "target_type", "document"
                            ),
                        )
                    )
                blossom_meta = {
                    "episode_id": f"offload:{getattr(parent_scorable, 'id', 'x')}",
                    "winners": [],
                    "top_reward": 0.0,
                    "k_requested": self.cfg.k_candidates,
                    "k_resolved": len(children),
                }

            else:
                # original synchronous path
                children, blossom_meta = await self._blossom_and_refine(
                    s,
                    {**context, "progress_task": ep_task},
                    self.cfg.k_candidates,
                )

            # best-effort progress reconciliation (if runner didn't emit live)
            self._reconcile_episode_progress_from_events(
                ep_task, blossom_meta, est_total
            )

            # end subtask
            self.pdone(ep_task)

            # diversity + novelty
            diversity_raw = self._blossom_diversity(children)
            filtered_children = self._novel_children(
                parent_scorable, children, tau=self.cfg.novelty_tau
            )
            diversity_post = self._blossom_diversity(filtered_children)

            # evaluate filtered
            cand_evals = await self._eval_many(
                filtered_children, context, use_vpm_phi
            )

            # NEW: cache per-child dim vectors globally
            for c, res in cand_evals:
                cid = str(getattr(c, "id", "") or "")
                if cid:
                    all_vectors[cid] = dict(res.get("dims") or {})

            # Optional: enrich top-K candidates by fused score (cheap guard)
            try:
                k_top = int(self.cfg.attr_enrichment.get("cands_topk", 0))
                if k_top > 0:
                    await self._enrich_candidates_topk(
                        cand_evals, k_top, context
                    )
            except Exception as _e:
                self._slog(
                    "AttrEnrichCandsTopKError", {"error": str(_e)}
                )

            # emit garden events for candidates
            for c, res in cand_evals:
                pid = getattr(
                    parent_scorable, "blossom_node_id", None
                ) or self._sid(parent_scorable, idx)
                nid = getattr(c, "blossom_node_id", None) or self._sid(
                    c, idx
                )
                self._emit_garden_event(
                    run_dir,
                    "add_node",
                    node_id=str(nid),
                    parent_id=str(pid),
                    overall=float(res.get("overall", 0.0)),
                    dims=res.get("dims", {}),
                    text_len=len(getattr(c, "text", "") or ""),
                    source="blossom",  # provenance
                )
                self._emit_garden_event(
                    run_dir,
                    "add_edge",
                    source=str(pid),
                    target=str(nid),
                    edge_type="candidate",
                )

            parent_overall = float(
                base_evals[self._sid(parent_scorable, idx)]
            )

            # compute merits then pick best
            parent_len = len(parent_scorable.text or "")
            best = None
            for c, res in cand_evals:
                m = self._merit(
                    base_vectors[
                        str(getattr(parent_scorable, "id", None))
                    ],
                    res,
                    text_len=len(c.text or ""),
                    base_len=parent_len,
                )
                if best is None or m > best[2]:
                    best = (c, res, m)
            if not best:
                # no viable children → keep parent
                winner, win_eval = (
                    parent_scorable,
                    {"overall": parent_overall, "dims": {}},
                )
            else:
                winner, win_eval = best[0], best[1]
                if (
                    float(win_eval.get("overall", 0.0)) - parent_overall
                    < self.cfg.promote_margin
                ):
                    winner, win_eval = (
                        parent_scorable,
                        {"overall": parent_overall, "dims": {}},
                    )

            # Track improved cohort metrics for Nexus view
            winner_sid = self._sid(winner, idx)
            win_overall = float((win_eval or {}).get("overall", parent_overall))
            improved_qualities[winner_sid] = win_overall
            improved_vectors[winner_sid] = dict((win_eval or {}).get("dims") or {})

            final_cohort.append(winner)  # <- collect improved cohort

            wid = getattr(winner, "blossom_node_id", None) or self._sid(
                winner, idx
            )
            pid = getattr(
                parent_scorable, "blossom_node_id", None
            ) or self._sid(parent_scorable, idx)

            winner_overall = float(
                (win_eval or {}).get("overall", parent_overall)
            )
            lift = float(winner_overall - parent_overall)
            if winner and lift >= self.cfg.promote_margin:
                self._emit_garden_event(
                    run_dir,
                    "promote",
                    parent_id=str(pid),
                    child_id=str(wid),
                    lift=float(lift),
                    child_overall=float(winner_overall),
                )
                # also mark node
                self._emit_garden_event(
                    run_dir,
                    "node_update",
                    node_id=str(wid),
                    status="promoted",
                    overall=float(winner_overall),
                )

            # blossom linkage summary for the decision
            decisions.append(
                {
                    "parent_id": getattr(parent_scorable, "id", None),
                    "parent_overall": parent_overall,
                    "winner_id": getattr(winner, "id", None),
                    "winner_overall": winner_overall,
                    "lift": lift,
                    "k_generated": len(children),
                    "k_after_novelty": len(filtered_children),
                    "diversity_raw": float(diversity_raw),
                    "diversity_post": float(diversity_post),
                    "blossom_episode_id": blossom_meta.get("episode_id"),
                    "blossom_winner_leaf_ids": [
                        w.get("leaf_id")
                        for w in blossom_meta.get("winners", [])
                    ],
                    "blossom_winner_rewards": [
                        float(w.get("reward", 0.0))
                        for w in blossom_meta.get("winners", [])
                    ],
                    "blossom_winner_paths": [
                        w.get("path")
                        for w in blossom_meta.get("winners", [])
                    ],
                    "blossom_top_reward": float(
                        blossom_meta.get("top_reward", 0.0)
                    ),
                }
            )

            self._emit_garden_event(
                run_dir,
                "decision",
                parent_id=str(pid),
                winner_id=str(wid) if wid is not None else None,
                lift=float(lift),
                k_generated=len(children),
                k_after_novelty=len(filtered_children),
            )

            # persist episode meta
            context["blossom_runs"].append(blossom_meta)

            # link & promote
            self._safe_link_and_promote(
                parent_scorable, [c for c, _ in cand_evals], winner, lift
            )

            # training events
            self._safe_emit_training(
                parent=parent_scorable,
                parent_overall=parent_overall,
                cand_evals=cand_evals,
                context=context,
            )

            # overall progress step
            self.pstep(task_run, n=1)

        # ---------------- 3) Cohort verdict ----------------
        self.pstage(task_run, stage="finalize")

        # NEW: winner-centric quality map for improved cohort
        improved_evals: Dict[str, float] = {}
        for d in decisions:
            wid = d.get("winner_id")
            if wid is None:
                continue
            improved_evals[str(wid)] = float(
                d.get("winner_overall", d.get("parent_overall", 0.0))
            )

        # ---------- (C) IMPROVED GRAPH ----------
        improved_manifest = self._manifest_from_scorables(run_id, final_cohort)
        improved_path = run_dir / "graph_improved.json"
        g2_report = self.graph_builder.build(
            run_id,
            improved_manifest,
            namespace="vpm:improved",
            knn_k=8,
            sim_threshold=0.35,
            add_temporal=True,
            add_mst_backbone=True,
            add_domain_edges=True,
            add_entity_edges=True,
            export_path=str(improved_path),
        )
        old_improved_path = run_dir / "old_graph_improved.json"
        self._write_graph(g2_report, old_improved_path)
        context["nexus_graph_json"] = str(
            improved_path
        )  # viewer now points at improved

        # NEW: sync improved cohort into Nexus graph
        if nexus_graph is not None:
            try:
                nexus_graph.ingest_manifest(
                    improved_manifest,
                    qualities=improved_evals,
                    metrics=all_vectors,
                    phase="improved",
                )
                if isinstance(g2_report, dict) and g2_report.get("graph_json"):
                    nexus_graph.write_edges_from_graph_json(
                        g2_report["graph_json"],
                        phase="improved",
                        channel="vpm:improved",
                    )
            except Exception as e:
                self._slog(
                    "NexusImprovedSyncError",
                    {"error": str(e)},
                )

        wins = sum(1 for d in decisions if d["lift"] > 0.0)
        win_rate = float(wins / max(1, len(decisions)))
        lifts = [float(d["lift"]) for d in decisions] or [0.0]
        cohort_lift = float(stats.mean(lifts))
        topk = sorted(base_evals.values(), reverse=True)[
            : max(1, math.ceil(0.1 * len(base_evals)))
        ]
        promoted_overalls = [
            max(d["parent_overall"], d["winner_overall"]) for d in decisions
        ]
        topk_new = sorted(promoted_overalls, reverse=True)[
            : max(1, math.ceil(0.1 * len(promoted_overalls)))
        ]
        topk_lift = (
            float((stats.mean(topk_new) - stats.mean(topk)))
            if topk and topk_new
            else 0.0
        )

        report = {
            "status": "ok",
            "goal_preview": goal_text[:120],
            "decision_count": len(decisions),
            "baseline": baseline_summary,
            "win_rate": win_rate,
            "mean_lift": cohort_lift,
            "topk_lift": topk_lift,
            "mean_blossom_diversity_raw": float(
                stats.mean([d["diversity_raw"] for d in decisions])
            )
            if decisions
            else 0.0,
            "mean_blossom_diversity_post": float(
                stats.mean([d["diversity_post"] for d in decisions])
            )
            if decisions
            else 0.0,
            "blossom_success_rate": self._blossom_success_rate(decisions),
            "decisions": decisions,
        }
        try:
            run_dir = self._run_dir(context)
            report_path = run_dir / "nexus_improver_report.json"
            report_path.write_text(
                dumps_safe(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            self._slog("WriteReportError", {"error": str(e)})
            report_path = None

        self.pdone(task_run)
        context[self.output_key] = report

        # Garden frames (may or may not return a path; tolerate None)
        frames_path = None
        try:
            frames_path = write_garden_frames(
                run_dir, baseline_graph_json=context.get("nexus_graph_json")
            )
        except Exception as e:
            self._slog("GardenFramesError", {"error": str(e)})
            frames_path = None

        # Knowledge index from baseline + improved graphs
        baseline_path = run_dir / "baseline_graph.json"
        improved_path = run_dir / "graph_improved.json"
        try:
            idx = compute_knowledge_index_from_files(
                baseline_path,
                improved_path,
                node_quality_attr="quality",
                edge_quality_attr="weight",
                extra_report=report_path,
            )
            (run_dir / "knowledge_index.json").write_text(
                dumps_safe(idx, indent=2), encoding="utf-8"
            )
            self._slog("KnowledgeIndex", idx)
        except Exception as e:
            self._slog("KnowledgeIndexError", {"error": str(e)})

        # ---------------- Nexus view manifest (NEW) ----------------
        try:
            # Baseline GraphScore: treat "quality" as baseline mean, and expose median/p90 too
            baseline_mean_dims = {
                "quality": float(baseline_summary.get("mean", 0.0)),
                "median": float(baseline_summary.get("median", 0.0)),
                "p90": float(baseline_summary.get("p90", 0.0)),
            }
            baseline_score_obj = GraphScore(
                node_count=len(baseline_manifest.get("items") or []),
                edge_count=int(g_report.get("edges_written", 0)),
                mean_dims=baseline_mean_dims,
            )

            # Improved GraphScore: use promoted_overalls + lift stats
            improved_mean_quality = float(
                stats.mean(promoted_overalls)
            ) if promoted_overalls else baseline_mean_dims["quality"]

            improved_mean_dims = {
                "quality": improved_mean_quality,
                "mean_lift": cohort_lift,
                "win_rate": win_rate,
                "topk_lift": topk_lift,
            }
            improved_score_obj = GraphScore(
                node_count=len(improved_manifest.get("items") or []),
                edge_count=int(g2_report.get("edges_written", 0)),
                mean_dims=improved_mean_dims,
            )

            view = NexusViewManifest(
                run_id=str(run_id),
                goal_preview=goal_text[:120],
                baseline_graph_path=str(baseline_path),
                improved_graph_path=str(improved_path),
                frames_path=str(frames_path) if frames_path else None,
                report_path=str(report_path) if report_path else None,
                baseline_score=baseline_score_obj.overall(),
                improved_score=improved_score_obj.overall(),
                extra={
                    "baseline_mean_dims": baseline_score_obj.mean_dims,
                    "improved_mean_dims": improved_score_obj.mean_dims,
                    "knowledge_index_path": str(run_dir / "knowledge_index.json"),
                },
            )

            view_path = run_dir / "nexus_view_manifest.json"
            view.write_json(view_path)
            context["nexus_view_manifest"] = str(view_path)
            self._slog("NexusViewManifest", {"path": str(view_path)})
        except Exception as e:
            self._slog("NexusViewManifestError", {"error": str(e)})

        return context

    # --------------------------- CORE HELPERS ---------------------------

    def _eval_state(
        self, scorable: Scorable, context: Dict[str, Any], *, use_vpm_phi: bool
    ) -> Dict[str, Any]:
        """
        Preferred path uses ScoringService.evaluate_state.

        Note: any enhanced row for this scorable is available at
              context["scorable_rows"].get(str(scorable.id)) if the
              ScoringService wants to peek at it.
        """
        try:
            return self.scoring.evaluate_state(
                scorable=scorable,
                context=context,
                scorers=self.cfg.scorers,
                dimensions=self.cfg.dims,
                scorer_weights=self.cfg.scorer_weights,
                dimension_weights=self.cfg.dimension_weights,
                include_llm_heuristic=self.cfg.use_llm_heuristic,
                include_vpm_phi=use_vpm_phi,
                fuse_mode="weighted_mean",
                clamp_01=True,
            )
        except Exception as e:
            self._slog(
                "EvaluateStateError",
                {"id": getattr(scorable, "id", None), "error": str(e)},
            )
            # Graceful but simple fallback
            return {
                "overall": 0.0,
                "dims": dict.fromkeys(self.cfg.dims or ["alignment"], 0.0),
                "by_scorer": {},
                "components": {},
            }

    async def _eval_many(
        self, cand: List[Scorable], context, use_vpm_phi: bool
    ):
        # Ensure each candidate also has a row
        if cand:
            tasks = [
                self.scorable_processor.process(c, context=context)
                for c in cand
            ]
            processed = await asyncio.gather(*tasks)
            rows_by_id = context.setdefault("scorable_rows", {})
            for c, row in zip(cand, processed):
                sid = str(getattr(c, "id", "") or "")
                if sid:
                    rows_by_id[sid] = row

        tasks = [
            asyncio.to_thread(
                lambda c=c: (
                    c,
                    self._eval_state(c, context, use_vpm_phi=use_vpm_phi),
                )
            )
            for c in cand
        ]
        return await asyncio.gather(*tasks)


    # inside NexusPollinatorAgent

    def _to_scorables(self, raw) -> List[Scorable]:
        out = []
        for x in raw or []:
            if isinstance(x, Scorable):
                out.append(x)
            else:
                out.append(Scorable.from_dict(x))
        return out

    def _manifest_from_scorables(
        self, run_id: Any, scorables: List[Scorable]
    ) -> Dict[str, Any]:
        items = []
        for s in scorables:
            # minimal, graph-friendly record
            items.append(
                {
                    "item_id": getattr(s, "id", None)
                    or getattr(s, "scorable_id", None),
                    "scorable_id": getattr(s, "id", None),
                    "scorable_type": getattr(s, "target_type", "document"),
                    "near_identity": {
                        "title": None,
                        "text": getattr(s, "text", None),
                    },
                    "domains": getattr(s, "domains", None),
                    "entities": getattr(s, "entities", None),
                    "chat_id": getattr(s, "chat_id", None),
                    "turn_index": getattr(s, "turn_index", None),
                    # Optional enrichments you already have available:
                    # "metrics_values": getattr(s, "metrics_values", None),
                    # "embeddings": {"global": getattr(s, "embed_global", None)},
                }
            )
        return {"run_id": str(run_id), "items": items}

    def _write_graph(self, report: Dict[str, Any], out_path: Path) -> None:
        try:
            gjson = report.get("graph_json")
            if gjson:
                out_path.write_text(
                    dumps_safe(gjson, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as e:
            self._slog(
                "WriteGraphError", {"path": str(out_path), "error": str(e)}
            )

    async def _blossom_and_refine(
        self, parent: Scorable, context: Dict[str, Any], k: int
    ) -> Tuple[List[Scorable], Dict[str, Any]]:
        goal_text = getattr(
            getattr(parent, "goal_ref", None) or {}, "text", None
        ) or (context.get("goal") or {}).get("goal_text", "")
        goal = self.memory.goals.get_or_create({"goal_text": goal_text})

        ep_task = context.get("progress_task")
        run_dir = self._run_dir(context)
        events_dir = run_dir / "blossom_events"
        events_dir.mkdir(parents=True, exist_ok=True)
        ev_path = events_dir / f"{str(getattr(parent, 'id', 'unknown'))}.jsonl"

        runner_ctx: Dict[str, Any] = {
            "goal": goal.to_dict(),
            "seed": {
                "seed_type": getattr(parent, "scorable_type", "scorable"),
                "seed_id": getattr(parent, "scorable_id", None),
                "plan_text": getattr(parent, "text", None),
            },
            "pipeline_run_id": context.get("pipeline_run_id"),
            "events_jsonl": str(ev_path),
            "progress_task": ep_task,
        }

        try:
            out_ctx = await self.blossom.run(runner_ctx)
        except Exception as e:
            self._slog(
                "BlossomRunError",
                {"parent": getattr(parent, "id", None), "error": str(e)},
            )
            return [], {}

        bres = (out_ctx or {}).get("blossom_result") or {}
        episode_id = bres.get("episode_id")
        winners = list(bres.get("winners") or [])
        training_batch = bres.get("training_batch")
        top_reward = float(winners[0]["reward"]) if winners else 0.0

        # resolve texts for each winner
        # resolve texts for each winner
        resolved_children: List[Scorable] = []
        resolved_winners: List[Dict[str, Any]] = []

        for i, w in enumerate(winners):
            text, resolved_node_id = self._winner_to_text(w)
            if not text:
                continue

            node_id = resolved_node_id

            # If we don't have a DB node for this text, create one so we get a real integer id
            if node_id is None:
                try:
                    if hasattr(self.memory, "blossoms") and hasattr(
                        self.memory.blossoms, "create_node"
                    ):
                        node = self.memory.blossoms.create_node(
                            blossom_id=int(episode_id)
                            if str(episode_id).isdigit()
                            else episode_id,
                            state_text=text,
                            # NB: If your schema expects "parent is a NODE id", you may need to resolve that separately.
                            # Using a scorable id here only if it's actually a node id in your system.
                            # parent_id=None  # safer default unless you truly have the node id
                        )
                        node_id = int(node.id)
                except Exception as e:
                    self._slog(
                        "BlossomCreateNodeError",
                        {"episode_id": episode_id, "error": str(e)},
                    )

            # Build the child once, with a stable id (prefer the real node id)
            scorable_id = (
                str(node_id)
                if isinstance(node_id, int)
                else f"blossom:{episode_id}:{resolved_node_id or i}"
            )
            child = Scorable(
                id=scorable_id,
                text=str(text),
                target_type=getattr(parent, "target_type", "document"),
            )
            # Persist the true DB node id (if any) for later linking/promote
            try:
                setattr(
                    child,
                    "blossom_node_id",
                    node_id if isinstance(node_id, int) else None,
                )
            except Exception:
                pass

            resolved_children.append(child)
            norm_path = self._normalize_winner_path(w.get("path"))
            resolved_winners.append(
                {
                    "leaf_id": w.get("leaf_id"),
                    "leaf_db_id": resolved_node_id,
                    "reward": float(w.get("reward", 0.0)),
                    "path": norm_path,
                    "sharpened_meta": (w.get("sharpened") or {}),
                    "text_len": len(text or ""),
                    "node_id": node_id if isinstance(node_id, int) else None,
                }
            )
            # Emit path edges (if any) and also store normalized paths in resolved_winners
            for rw in resolved_winners:
                raw_path = next(
                    (
                        w.get("path")
                        for w in winners
                        if str(w.get("leaf_id")) == str(rw.get("leaf_id"))
                    ),
                    None,
                )
                norm_path = self._normalize_winner_path(raw_path)
                rw["path"] = (
                    norm_path  # keep the normalized version for the film/meta
                )

                # Only emit if we have a chain of at least 2 nodes
                if len(norm_path) >= 2:
                    for u, v in zip(norm_path, norm_path[1:]):
                        try:
                            self._emit_garden_event(
                                run_dir,
                                "add_edge",
                                source=str(u),
                                target=str(v),
                                edge_type="path",
                            )
                        except Exception as _e:
                            self._slog(
                                "BlossomPathEdgeEmitError",
                                {"u": u, "v": v, "error": str(_e)},
                            )

        blossom_meta = {
            "episode_id": episode_id,
            "parent_id": getattr(parent, "id", None),
            "parent_text_len": len(getattr(parent, "text", "") or ""),
            "goal_text": (context.get("goal") or {}).get("goal_text", ""),
            "k_requested": int(k),
            "k_resolved": len(resolved_children),
            "top_reward": top_reward,
            "winners": resolved_winners,
            "training_batch_present": training_batch is not None,
            "events_jsonl": (out_ctx or {}).get("events_jsonl")
            or str(ev_path),
            "runner_cfg": {
                "return_top_k": getattr(self.blossom.cfg, "return_top_k", None)
                if hasattr(self.blossom, "cfg")
                else None,
                "sharpen_top_k": getattr(
                    self.blossom.cfg, "sharpen_top_k", None
                )
                if hasattr(self.blossom, "cfg")
                else None,
            },
        }

        return resolved_children, blossom_meta

    def _winner_to_text(
        self, w: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Best-effort recovery of winner text:
          1) prefer w["sharpened"]["sharpened"]
          2) else try BlossomStore.get_node(leaf_db_id)
          3) else try BlossomStore.get_node(int(leaf_id)) if numeric
        Returns (text, resolved_node_id).
        """
        sh = w.get("sharpened") or {}
        if isinstance(sh, dict) and sh.get("sharpened"):
            return str(sh["sharpened"]), None

        node_id = w.get("leaf_db_id")
        try_ids: List[int] = []
        if node_id is not None:
            try:
                try_ids.append(int(node_id))
            except Exception:
                pass

        lid = w.get("leaf_id")
        if lid is not None:
            try:
                try_ids.append(int(lid))
            except Exception:
                pass

        for nid in try_ids:
            try:
                node = self.memory.blossoms.get_node(int(nid))
                if node:
                    text = node.sharpened_text or node.state_text
                    if text:
                        return str(text), int(nid)
            except Exception as e:
                self._slog(
                    "BlossomFetchNodeError",
                    {"leaf_candidate_id": nid, "error": str(e)},
                )

        return None, None

    def _novel_children(
        self, parent: Scorable, children: List[Scorable], tau: float
    ) -> List[Scorable]:
        """
        Keep children whose cosine distance from parent >= tau.
        If everything is filtered, keep the single best child (preserve progress).
        """
        if not children:
            return []
        try:
            pe = np.asarray(
                self.memory.embedding.get_or_create(parent.text), dtype=float
            )
            pn = float(np.linalg.norm(pe)) + 1e-9
            keep: List[Scorable] = []
            sims: List[Tuple[float, Scorable]] = []
            for c in children:
                ce = np.asarray(
                    self.memory.embedding.get_or_create(c.text), dtype=float
                )
                cn = float(np.linalg.norm(ce)) + 1e-9
                sim = float(np.dot(pe, ce) / (pn * cn))
                sims.append((sim, c))
                if (1.0 - sim) >= tau:
                    keep.append(c)
            if keep:
                return keep
            sims.sort(key=lambda t: t[0])  # ascending by similarity
            return [sims[0][1]]
        except Exception:
            return children

    def _persist_dims(
        self,
        scorable: Scorable,
        eval_res: Dict[str, Any],
        context: Dict[str, Any],
    ):
        try:
            self.scoring.save_hrm_score(
                scorable_id=scorable.id,
                scorable_type=scorable.target_type,
                value=float(eval_res["overall"]),
                **{
                    k: context.get(k)
                    for k in ("goal_id", "plan_trace_id", "pipeline_run_id")
                    if k in context
                },
            )
            for d, v in (eval_res.get("dims") or {}).items():
                self.scoring.save_score(
                    scorable_id=scorable.id,
                    scorable_type=scorable.target_type,
                    score_type=str(d),
                    score_value=float(v),
                    source="state_evaluator",
                    **{
                        k: context.get(k)
                        for k in (
                            "goal_id",
                            "plan_trace_id",
                            "pipeline_run_id",
                        )
                        if k in context
                    },
                )
        except Exception as e:
            self._slog(
                "PersistDimsError",
                {"id": getattr(scorable, "id", None), "error": str(e)},
            )

    def _sid(self, s: Scorable, idx: Optional[int] = None) -> str:
        v = getattr(s, "id", None)
        if v is None:
            return f"tmp-{idx if idx is not None else 'x'}"
        return str(v)

    def _merit(
        self,
        parent_vec,
        child_res,
        *,
        text_len,
        base_len,
        advantage_w=0.25,
        novelty_w=0.15,
        delta_bonus=0.0,
    ):
        overall = float(child_res.get("overall", 0.0))
        dims = child_res.get("dims", {}) or {}
        clarity = float(dims.get("clarity", overall))
        faithful = float(dims.get("faithfulness", overall))
        len_ratio = max(1.0, float(text_len) / max(1.0, float(base_len)))
        length_penalty = 0.08 * math.log(len_ratio)

        # Optional novelty: if present in child_res
        novelty = float(child_res.get("novelty", 0.0))
        # Optional advantage: if present in child_res
        advantage = float(child_res.get("advantage", 0.0))

        merit = (
            0.55 * overall
            + 0.2 * clarity
            + 0.2 * faithful
            + advantage_w * advantage
            + novelty_w * novelty
            - length_penalty
            + delta_bonus
        )
        log.info(
            f"Merit calc: overall={overall:.4f}, clarity={clarity:.4f}, faithful={faithful:.4f}, "
            f"advantage={advantage:.4f}, novelty={novelty:.4f}, length_penalty={length_penalty:.4f}, "
            f"delta_bonus={delta_bonus:.4f} => merit={merit:.44f}"
        )
        return merit

    def _has_rich_attrs(self, scorable_id: str) -> bool:
        """Check if scorable already has 'enough' metrics to skip enrichment."""
        min_cols = int(self.cfg.attr_enrichment.get("min_columns", 120))
        try:
            # Prefer DAO if present
            if hasattr(self.memory, "nexus") and hasattr(self.memory.nexus, "get_metrics"):
                m = self.memory.nexus.get_metrics(scorable_id)
                if m and isinstance(m.columns, (list, tuple)) and len(m.columns) >= min_cols:
                    return True
                # Some stores expose pre-materialized vector dict
                if m and isinstance(getattr(m, "vector", None), dict) and len(m.vector) >= min_cols:
                    return True
        except Exception:
            pass
        # In-run cache
        return scorable_id in self._enriched_ids

    async def _maybe_enrich(self, scorable: Scorable, context: Dict[str, Any], *, when: str) -> bool:
        """JIT enrichment with budget + parallelism guard. Returns True if run (or already enriched)."""
        if not self.cfg.attr_enrichment.get("enabled", True):
            return False

        sid = str(getattr(scorable, "id", "") or "")
        if not sid or self._has_rich_attrs(sid):
            return False
        if self._attr_budget <= 0:
            return False

        enhanced = scorable  # keep simple; your processor already accepts Scorable + context

        try:
            async with self._attr_inflight:
                # Consume budget *before* work to avoid stampede on low budgets
                self._attr_budget -= 1
                # Run potentially heavy sync work off the main loop
                row = await self.scorable_processor.process(
                    enhanced,
                    context=context,
                )
                self._enriched_ids.add(sid)
                self._slog(
                    "AttrEnrichOK",
                    {
                        "id": sid,
                        "when": when,
                        "columns": getattr(
                            getattr(row, "metrics", None), "columns", None
                        )
                        or "unknown",
                    },
                )
                return True
        except Exception as e:
            self._slog(
                "AttrEnrichError",
                {"id": sid, "when": when, "error": str(e)},
            )
            return False

    async def _enrich_topq_parents(self, items: list[Scorable], base_evals: dict[str, float], q: float, context: dict):
        """Enrich parents whose baseline overall is in top quantile q."""
        if not items or q is None:
            return
        try:
            vals = sorted(base_evals.values())
            if not vals:
                return
            import math
            qi = max(0, min(len(vals)-1, int(math.floor(q * (len(vals)-1)))))
            thresh = vals[qi]
        except Exception:
            # fall back to mean if quantile math fails
            thresh = float(sum(base_evals.values())/max(1, len(base_evals)))

        tasks = []
        for s in items:
            sid = str(getattr(s, "id", "") or "")
            if sid and base_evals.get(sid, 0.0) >= thresh:
                tasks.append(self._maybe_enrich(s, context, when="baseline_topq"))
        if tasks:
            await asyncio.gather(*tasks)

    async def _enrich_candidates_topk(self, cand_evals: list[tuple[Scorable, dict]], k: int, context: dict):
        """Optionally enrich top-K candidates by fused overall."""
        if k <= 0 or not cand_evals:
            return
        top = sorted(cand_evals, key=lambda t: float(t[1].get("overall", 0.0)), reverse=True)[:k]
        await asyncio.gather(*[self._maybe_enrich(s, context, when="cands_topk") for s, _ in top])

    def _safe_link_and_promote(
        self,
        parent: Scorable,
        candidates: List[Scorable],
        winner: Optional[Scorable],
        lift: float,
    ):
        def _to_int(x) -> Optional[int]:
            try:
                if isinstance(x, int):
                    return x
                if isinstance(x, str) and x.isdigit():
                    return int(x)
            except Exception:
                pass
            return None

        # link parent->children using only integer node ids
        try:
            int_child_ids: List[int] = []
            for c in candidates:
                nid = getattr(c, "blossom_node_id", None)
                if not isinstance(nid, int):
                    nid = _to_int(getattr(c, "id", None))
                if isinstance(nid, int):
                    int_child_ids.append(nid)

            pid = getattr(parent, "blossom_node_id", None)
            if not isinstance(pid, int):
                pid = _to_int(getattr(parent, "id", None))

            if (
                int_child_ids
                and pid is not None
                and hasattr(self.memory, "blossoms")
                and hasattr(self.memory.blossoms, "link_parent_children")
            ):
                self.memory.blossoms.link_parent_children(
                    parent_id=pid, child_ids=int_child_ids
                )
        except Exception as e:
            self._slog(
                "BlossomLinkError",
                {"parent": getattr(parent, "id", None), "error": str(e)},
            )

        # promote only if winner maps to a real integer node id
        try:
            if (
                winner
                and lift >= self.cfg.promote_margin
                and hasattr(self.memory, "nexus")
                and hasattr(self.memory.nexus, "promote")
            ):
                wid = getattr(winner, "blossom_node_id", None)
                if not isinstance(wid, int):
                    wid = _to_int(getattr(winner, "id", None))
                pid = getattr(parent, "blossom_node_id", None)
                if not isinstance(pid, int):
                    pid = _to_int(getattr(parent, "id", None))
                if isinstance(wid, int) and isinstance(pid, int):
                    self.memory.nexus.promote(
                        parent_id=pid, child_id=wid, reason="local_improvement"
                    )
        except Exception as e:
            self._slog(
                "NexusPromoteError",
                {"parent": getattr(parent, "id", None), "error": str(e)},
            )

    def _safe_emit_training(
        self,
        *,
        parent: Scorable,
        parent_overall: float,
        cand_evals: List[Tuple[Scorable, Dict[str, Any]]],
        context: Dict[str, Any],
    ):
        if not hasattr(self.memory, "training_events"):
            return
        for c, res in cand_evals:
            try:
                child_overall = float(res.get("overall", 0.0))
                prefer_child = child_overall > parent_overall
                pos, neg = (c, parent) if prefer_child else (parent, c)
                w = max(
                    0.1, min(1.0, abs(child_overall - parent_overall) + 0.2)
                )

                # pairwise
                self.memory.training_events.insert_pairwise(
                    {
                        "model_key": "ranker.sicql.v1",
                        "dimension": "alignment",
                        "query_text": (context.get("goal") or {}).get(
                            "goal_text", ""
                        ),
                        "pos_text": pos.text,
                        "neg_text": neg.text,
                        "weight": w,
                        "trust": w * 0.6,
                        "goal_id": (context.get("goal") or {}).get("id"),
                        "pipeline_run_id": context.get("pipeline_run_id"),
                        "agent_name": "NexusImproverAgent",
                        "source": "nexus_improver",
                        "meta": {
                            "delta_overall": child_overall - parent_overall
                        },
                    },
                    dedup=True,
                )

                # pointwise (both)
                for s, v in ((c, child_overall), (parent, parent_overall)):
                    self.memory.training_events.insert_pointwise(
                        {
                            "model_key": "retriever.mrq.v1",
                            "dimension": "alignment",
                            "query_text": (context.get("goal") or {}).get(
                                "goal_text", ""
                            ),
                            "cand_text": s.text,
                            "label": 1 if v >= parent_overall else 0,
                            "weight": max(0.1, float(v)),
                            "trust": 0.5,
                            "goal_id": (context.get("goal") or {}).get("id"),
                            "pipeline_run_id": context.get("pipeline_run_id"),
                            "agent_name": "NexusImproverAgent",
                            "source": "nexus_improver",
                        },
                        dedup=True,
                    )
            except Exception as e:
                self._slog(
                    "TrainingEmitError",
                    {
                        "parent": parent.id,
                        "child": getattr(c, "id", None),
                        "error": str(e),
                    },
                )

    def _emit_garden_event(self, run_dir: Path, kind: str, **data):
        rec = {"ts": int(time.time()), "kind": kind, **data}
        try:
            with (run_dir / "garden_events.jsonl").open(
                "a", encoding="utf-8"
            ) as f:
                f.write(dumps_safe(rec, ensure_ascii=False) + "\n")
                f.flush()
        except Exception:
            pass

    # --------------------------- METRICS / UTIL ---------------------------

    def _blossom_diversity(self, children: List[Scorable]) -> float:
        if len(children) < 2:
            return 0.0
        try:
            embs = [
                np.asarray(
                    self.memory.embedding.get_or_create(c.text), dtype=float
                )
                for c in children
            ]
            embs = [e / (np.linalg.norm(e) + 1e-9) for e in embs]
            sims = []
            for i in range(len(embs)):
                ei = embs[i]
                for j in range(i + 1, len(embs)):
                    sims.append(float(np.dot(ei, embs[j])))
            return float(max(0.0, 1.0 - (sum(sims) / max(1, len(sims)))))
        except Exception:
            return 0.0

    def _blossom_success_rate(self, decisions: List[Dict[str, Any]]) -> float:
        ok = sum(1 for d in decisions if float(d.get("lift", 0.0)) > 0.0)
        return float(ok / max(1, len(decisions)))

    def _cohort_summary(self, vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        vals_sorted = sorted(vals)
        p90 = vals_sorted[max(0, int(0.9 * (len(vals_sorted) - 1)))]
        return {
            "mean": float(stats.mean(vals_sorted)),
            "median": float(stats.median(vals_sorted)),
            "p90": float(p90),
        }

    def _reconcile_episode_progress_from_events(
        self, ep_task: str, blossom_meta: Dict[str, Any], est_total: int
    ) -> None:
        """
        If the runner didn't emit live progress, tail the JSONL and set a sane final count.
        """
        try:
            events_path = blossom_meta.get("events_jsonl")
            if not events_path:
                return
            p = Path(events_path)
            if not p.exists():
                return
            made = 0
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        if j.get("kind") == "add_node":
                            made += 1
                    except Exception:
                        continue
            if made:
                self.ptick(ep_task, done=made, total=max(made, est_total))
        except Exception:
            # best-effort only
            pass

    def _normalize_winner_path(self, path_val) -> List[str]:
        """
        Accepts None | list[int|str|dict] | str and returns a simple list[str] node ids.
        Dict items may contain keys like 'node_id' | 'id' | 'leaf_id'.
        Strings like '1->2->3' or '1,2,3' are supported.
        """
        if not path_val:
            return []

        # List or tuple of mixed items
        if isinstance(path_val, (list, tuple)):
            out: List[str] = []
            for item in path_val:
                if isinstance(item, (int, str)):
                    s = str(item).strip()
                    if s:
                        out.append(s)
                elif isinstance(item, dict):
                    for k in ("node_id", "id", "leaf_id"):
                        if k in item and item[k] is not None:
                            s = str(item[k]).strip()
                            if s:
                                out.append(s)
                            break
            return out

        # String form: "1->2->3" or "1,2,3" or whitespace separated
        if isinstance(path_val, str):
            import re

            toks = re.split(r"\s*(?:->|,|\s)\s*", path_val.strip())
            return [t for t in toks if t]

        return []

    def _task_run_name(self, context: Dict[str, Any]) -> str:
        rid = context.get("pipeline_run_id") or int(time.time())
        return f"nexus_pollinator:{rid}"

    def _task_episode_name(self, parent_id: Any) -> str:
        return f"blossom_episode:parent={parent_id}"

    def _run_dir(self, context: Dict[str, Any]) -> Path:
        """
        Resolve a run directory suitable for logs/artifacts.
        """
        rid = context.get("pipeline_run_id") or int(time.time())
        root = Path(self.cfg.blossom_cfg.get("run_root", "runs"))
        d = root / f"run-{rid}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _normalize_weights(
        self, w: Dict[str, float], default: float = 1.0
    ) -> Dict[str, float]:
        w2 = {k: float(v) for k, v in (w or {}).items()}
        if not w2:
            return {}
        s = sum(abs(v) for v in w2.values())
        return {k: (v / s if s else default) for k, v in w2.items()}

    def _get_nexus_graph(self, context: Dict[str, Any]) -> Optional[NexusGraph]:
        """
        Best-effort: return a run-scoped NexusGraph wrapper if a Nexus store is
        available on memory (memory.nexus or memory.nexus_store).
        """
        run_id = context.get("pipeline_run_id")
        if not run_id:
            return None

        store = getattr(self.memory, "nexus", None) or getattr(
            self.memory, "nexus_store", None
        )
        if store is None:
            return None

        try:
            cfg = NexusGraphConfig(
                run_id=str(run_id),
                namespace="nexus_pollinator",
            )
            return NexusGraph(cfg=cfg, memory=self.memory, container=self.container, logger=self.logger, run_id=str(run_id))
        except Exception as e:
            self._slog("NexusGraphInitError", {"error": str(e)})
            return None

    # ---- Async prompt helpers ---------------------------------
    async def _offload_candidate_prompts(
        self,
        parent: Scorable,
        context: Dict[str, Any],
        *,
        k: int,
        model: str,
        target_pool: str,
    ) -> List[Dict[str, str]]:
        """
        Enqueue K refinement prompts for `parent` via async prompt tool.
        Returns tickets: [{job_id, return_topic, parent_id, k_index}, ...]
        Falls back to [] on publish errors (caller will handle fallback).
        """

        goal_text = (context.get("goal") or {}).get("goal_text", "")
        prompts = []
        for i in range(k):
            prompts.append({
                "messages": [
                    {"role": "system", "content": "You are a careful refiner."},
                    {"role": "user", "content":
                        f"Refine this answer for the goal:\n{goal_text}\n\n---\n{parent.text}\n"}
                ],
            })

        client = PromptClient(self.cfg.blossom_cfg, self.memory, self.container, self.logger)

        # Coerce priority to enum (you were passing a string)
        prio = Priority.high if hasattr(Priority, "high") else Priority.normal

        try:
            tickets_raw = await client.offload_many(
                scorable_id=str(parent.id),
                prompts=prompts,
                model=model,
                target_pool=target_pool,
                priority=prio,
                group_key=f"nexus:{context.get('pipeline_run_id')}",
                meta={"purpose": "nexus_blossom_refine", "parent_id": str(parent.id)},
                response_format="text",
            )
        except Exception as e:
            self._slog("PromptOffloadError", {"parent": getattr(parent, "id", None), "error": str(e)})
            return []

        # Normalize & log
        tickets = [
            {"job_id": jid, "return_topic": rt, "parent_id": str(parent.id), "k_index": str(i)}
            for i, (jid, rt) in enumerate(tickets_raw or [])
            if jid and rt
        ]
        log.info("Prompt offload: requested=%d got=%d backend=%s",
                k, len(tickets), getattr(self.memory.bus, "get_backend", lambda: "unknown")())

        return tickets

    async def _await_offloaded_candidates(
        self,
        tickets: List[Dict[str, str]],
        *,
        timeout_s: float,
        poll_ms: int,
    ) -> List[str]:
        """
        Wait for text results for given tickets. Returns list[str] texts (may be shorter on timeout).
        Uses PromptClient if it exposes a wait API; otherwise does a simple poll loop as fallback.
        """

        client = PromptClient(
            self.cfg.blossom_cfg, self.memory, self.container, self.logger
        )

        # Fast-path: client has a vectorized wait
        return await client.wait_many(tickets, timeout_s=timeout_s)

    async def _materialize_rows(
        self,
        items: List[Scorable],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run ScorableProcessor on all parents once up-front and stash the
        enhanced 'rows' into context["scorable_rows"].

        Returns: {scorable_id: row}
        """
        if not items:
            return {}

        # Run Scorpions / ScorableProcessor in parallel
        tasks = [
            self.scorable_processor.process(s, context=context)
            for s in items
        ]
        processed = await asyncio.gather(*tasks)

        rows_by_id: Dict[str, Any] = {}
        for s, row in zip(items, processed):
            sid = str(getattr(s, "id", "") or "")
            if sid:
                rows_by_id[sid] = row

        # Make rows visible to downstream components
        ctx_rows = context.setdefault("scorable_rows", {})
        ctx_rows.update(rows_by_id)
        return rows_by_id

    def _slog(self, event: str, payload: Dict[str, Any]):
        try:
            if hasattr(self.logger, "log"):
                self.logger.log(event, payload)
            else:
                logging.getLogger(__name__).warning("%s: %s", event, payload)
        except Exception:
            pass
