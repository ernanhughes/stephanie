# stephanie/components/nexus/agents/nexus_inline.py
from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageSequence

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.nexus.app.manifest import (
    ManifestItem,
    NexusRunManifest,
)
from stephanie.components.nexus.graph.builder import (
    build_edges_enhanced,
    build_nodes_from_manifest,
)
from stephanie.components.nexus.graph.exporters import (
    export_graph_json,
    export_pyvis_html,
    export_pyvis_html_rich,
)
from stephanie.components.nexus.graph.layout import compute_positions
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.scoring_service import ScoringService
from stephanie.services.workers.nexus_workers import (
    NexusMetricsWorkerInline,
    NexusVPMWorkerInline,
)
from stephanie.services.zeromodel_service import ZeroModelService
from stephanie.utils.embed_utils import as_list_floats, cos_safe, has_vec
from stephanie.utils.graph_utils import compute_run_metrics
from stephanie.utils.json_sanitize import dumps_safe
from stephanie.utils.progress_mixin import ProgressMixin

log = logging.getLogger(__name__)


# ultra-light timing context
@contextmanager
def _stage(logger, name: str, **meta):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        payload = {"stage": name, "ms": round(dt_ms, 2)}
        payload.update(meta or {})
        try: 
            log.info("StageTiming %s", payload)
        except Exception:
            log.info("StageTiming %s: %.2f ms %s", name, dt_ms, meta or {})


def _first_frame_from_gif(gif_path: Path) -> Optional[Image.Image]:
    try:
        im = Image.open(gif_path)
        for frame in ImageSequence.Iterator(im):
            return frame.convert("RGB")
    except Exception:
        return None
    return None


def _find_tile_path(item_dir: Path) -> Optional[Path]:
    """
    Find a representative VPM tile for an item:
    1) any vpm*.png anywhere under item_dir (handles vpm_gray.png, vpm.png, etc.)
    2) fallback: first frame of filmstrip.gif saved to item_dir/vpm_from_gif.png
    """
    # 1) recursive search for vpm*.png
    for p in item_dir.rglob("vpm*.png"):
        if p.is_file():
            return p

    # 2) fallback to per-item filmstrip.gif (if present)
    gif = next((p for p in item_dir.rglob("filmstrip.gif") if p.is_file()), None)
    if gif:
        fr = _first_frame_from_gif(gif)
        if fr:
            out = item_dir / "vpm_from_gif.png"
            try:
                fr.save(out)
                return out
            except Exception:
                return None
    return None


def _consensus_walk_order(
    scorables: list[dict],
    *,
    goal_vec,
    alpha: float = 0.7,
    limit: int | None = None,
) -> list[int]:
    """
    Greedy run ordering that 'thinks':
      pick an item, update consensus (centroid of picked), then pick the next
      that maximizes: alpha*sim(goal) + (1-alpha)*sim(consensus).

    Safe for numpy arrays / mixed vector shapes. Falls back to metrics_vector if no embedding.
    """
    # --- collect vectors safely (no ndarray truthiness) ---
    vecs: list[list[float]] = []
    for s in scorables:
        v = as_list_floats((s.get("embeddings") or {}).get("global"))
        if not v:
            mv = s.get("metrics_vector")
            if isinstance(mv, dict):
                v = as_list_floats(list(mv.values()))
        vecs.append(v)

    n = len(scorables)
    if n == 0:
        return []

    tgt = as_list_floats(goal_vec)

    # If absolutely no vectors, keep stable order up to limit
    if not any(vecs):
        want = limit or n
        return list(range(min(want, n)))

    # --- start at argmax sim to goal (ties stable) ---
    sims_to_goal = [(cos_safe(v, tgt), i) for i, v in enumerate(vecs)]
    start_idx = max(sims_to_goal, key=lambda t: (t[0], -t[1]))[1]

    picked: list[int] = [start_idx]
    picked_set = {start_idx}

    # running centroid (len = dim of first non-empty picked vector)
    def centroid(indices: list[int]) -> list[float]:
        for i in indices:
            if vecs[i]:
                d = len(vecs[i])
                break
        else:
            return []
        acc = [0.0] * d
        cnt = 0
        for i in indices:
            v = vecs[i]
            if not v:
                continue
            cnt += 1
            # tolerate length mismatch: add up to min(d, len(v))
            m = min(d, len(v))
            for k in range(m):
                acc[k] += v[k]
        if cnt == 0:
            return []
        return [x / cnt for x in acc]

    c = centroid(picked)
    want = limit or n

    while len(picked) < want:
        best_score = -1e9
        best_j = None
        for j in range(n):
            if j in picked_set:
                continue
            v = vecs[j]
            s_goal = cos_safe(v, tgt)
            s_cons = cos_safe(v, c) if c else 0.0
            score = alpha * s_goal + (1.0 - alpha) * s_cons
            if score > best_score:
                best_score, best_j = score, j
        if best_j is None:
            break
        picked.append(best_j)
        picked_set.add(best_j)
        c = centroid(picked)

    return picked


class NexusInlineAgent(BaseAgent, ProgressMixin):
    """
    Builds a Nexus run (manifest → graph → frames) and, when enabled,
    produces A/B comparison runs:
      - baseline: random sample (size = top_k)
      - targeted: top_k by cosine similarity to the goal embedding
    Similarity is computed in text space using embeddings.global (or item text).
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        self.zm: ZeroModelService = self.container.get("zeromodel")
        self.scoring: ScoringService = self.container.get("scoring")
        self.scorers = cfg.get("scorers", ["sicql", "hrm", "tiny"])
        self.dimensions = cfg.get(
            "dimensions",
            ["alignment", "clarity", "relevance", "coverage", "faithfulness"],
        )

        self.vpmw = NexusVPMWorkerInline(self.zm, logger=logger)
        self.mxw = NexusMetricsWorkerInline(
            scoring=self.scoring,
            scorers=self.scorers,
            dimensions=self.dimensions,
            persist=False,
        )

        self.vpm_out = self.cfg.get("vpm_out", "./runs/nexus_vpm/")
        self.rollout_steps = int(self.cfg.get("rollout_steps", 0))
        self.rollout_strategy = self.cfg.get(
            "rollout_strategy", "consensus-walk"
        )
        self.target_type = self.cfg.get(
            "target_type", ScorableType.CONVERSATION_TURN
        )
        self.use_ab = bool(self.cfg.get("ab_compare", {}).get("enabled", True))

        # Optional toggles (default True)
        self.enable_html = bool(
            self.cfg.get("graph", {}).get("export_html", True)
        )
        self.enable_rich = bool(
            self.cfg.get("graph", {}).get("export_html_rich", True)
        )
        self.enable_filmstrip = bool(
            self.cfg.get("vpm", {}).get("filmstrip_enabled", True)
        )

        # Snapshot file handle (set per run)
        self._snap_path: Optional[Path] = None

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # progress + one-time init
        self._init_progress(self.container, self.logger)
        with _stage(self.logger, "ZeroModel.initialize"):
            self.zm.initialize()

        run_id_root = context.get(PIPELINE_RUN_ID)

        # If both sets exist, emit two runs and return
        if context.get("scorables_targeted") and context.get(
            "scorables_baseline"
        ):
            # ensure vectors exist (normalizes numpy etc.)
            await self._ensure_embeddings_global(context["scorables_targeted"])
            await self._ensure_embeddings_global(context["scorables_baseline"])

            # build a goal vector (same source as the A/B split)
            gx = (context.get("goal") or {}).get("goal_text") or ""
            target_vec = as_list_floats(self.memory.embedding.get_or_create(gx)) if gx else []
            tgt_ctx = await self._emit_single_run(
                run_id=f"{run_id_root}-targeted",
                scorables=context["scorables_targeted"],
                context=context,
                target_vec=target_vec,
            )
            bl_ctx = await self._emit_single_run(
                run_id=f"{run_id_root}-baseline",
                scorables=context["scorables_baseline"],
                context=context,
                target_vec=target_vec,
            )
            # publish A/B paths for GAP
            context["ab_targeted_run_dir"] = tgt_ctx["nexus_run_dir"]
            context["ab_baseline_run_dir"] = bl_ctx["nexus_run_dir"]
            return context

        scorables: List[Dict[str, Any]] = list(context.get("scorables", []))

        # Ensure each scorable has embeddings.global (computed from text if missing)
        with _stage(
            self.logger, "Nexus.ensure_embeddings", count=len(scorables)
        ):
            task = f"nexus:embeds:{context.get(PIPELINE_RUN_ID, 'na')}"
            self.pstart(
                task=task,
                total=len(scorables),
                meta={"phase": "ensure_embeddings"},
            )
            await self._ensure_embeddings_global(
                scorables,
                progress=lambda i, n: self.ptick(task=task, done=i, total=n),
            )
            self.pdone(task=task)

        # Build a goal/target vector for similarity trace & A/B selection
        goal_text = context.get("goal", {}).get("goal_text", "")
        target_vec = as_list_floats(
            self.memory.embedding.get_or_create(goal_text)
        )

        # A/B mode: plan two lists of scorables first, then emit two runs
        if self.use_ab:
            ab_cfg = self.cfg.get("ab_compare") or {}
            top_k = int(ab_cfg.get("top_k", 40))
            seed = int(ab_cfg.get("seed", 0))

            with _stage(
                self.logger, "Nexus.ab_plan", k=top_k, n=len(scorables)
            ):
                task_plan = (
                    f"nexus:ab_plan:{context.get(PIPELINE_RUN_ID, 'na')}"
                )
                self.pstart(
                    task=task_plan,
                    total=1,
                    meta={"k": top_k, "n": len(scorables)},
                )
                baseline_ids, targeted_ids = self._plan_ab_from_scorables(
                    scorables, target_vec, top_k=top_k, seed=seed
                )
                self.ptick(task=task_plan, done=1, total=1)
                self.pdone(task=task_plan)

            baseline_scorables = [scorables[i] for i in baseline_ids]
            targeted_scorables = [scorables[i] for i in targeted_ids]

            # Emit baseline
            base_ctx = await self._emit_single_run(
                run_id=f"{run_id_root}-baseline",
                scorables=baseline_scorables,
                context=context,
                target_vec=target_vec,
            )

            # Emit targeted
            tgt_ctx = await self._emit_single_run(
                run_id=f"{run_id_root}-targeted",
                scorables=targeted_scorables,
                context=context,
                target_vec=target_vec,
            )

            # Return both paths
            context.update(
                {
                    "ab_baseline_run_id": Path(base_ctx["nexus_run_dir"]).name,
                    "ab_targeted_run_id": Path(tgt_ctx["nexus_run_dir"]).name,
                    "ab_baseline_run_dir": base_ctx["nexus_run_dir"],
                    "ab_targeted_run_dir": tgt_ctx["nexus_run_dir"],
                }
            )

            # Compare A/B and write a single report at the parent dir
            try:
                base_dir = Path(base_ctx["nexus_run_dir"])
                tgt_dir = Path(tgt_ctx["nexus_run_dir"])
                parent = base_dir.parent  # common root e.g., runs/nexus_vpm

                def _load(p):
                    with open(
                        p / "run_metrics.json", "r", encoding="utf-8"
                    ) as f:
                        return json.load(f)

                mb = _load(base_dir)
                mt = _load(tgt_dir)

                def diff(a, b, key, subkey=None):
                    va = a[key][subkey] if subkey else a[key]
                    vb = b[key][subkey] if subkey else b[key]
                    # report improvement where "higher is better" except mean_edge_len (lower is better)
                    if key == "spatial" and subkey == "mean_edge_len":
                        return (va - vb) / max(1e-9, va)  # lower is better
                    return (vb - va) / max(1e-9, abs(va) + 1e-9)

                report = {
                    "baseline_id": str(base_dir.name),
                    "targeted_id": str(tgt_dir.name),
                    "improvements": {
                        "goal_alignment.mean": diff(
                            mb, mt, "goal_alignment", "mean"
                        ),
                        "goal_alignment.p90": diff(
                            mb, mt, "goal_alignment", "p90"
                        ),
                        "mutual_knn_frac": diff(mb, mt, "mutual_knn_frac"),
                        "clustering_coeff": diff(mb, mt, "clustering_coeff"),
                        "spatial.mean_edge_len": diff(
                            mb, mt, "spatial", "mean_edge_len"
                        ),
                        "largest_component": diff(mb, mt, "largest_component"),
                        "avg_degree": diff(mb, mt, "avg_degree"),
                        "mean_edge_weight": diff(mb, mt, "mean_edge_weight"),
                    },
                    "baseline": mb,
                    "targeted": mt,
                }
                ab_report_path = parent / f"{run_id_root}_ab_compare.json"
                with ab_report_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log.info(
                    "A/B compare written to %s", ab_report_path.as_posix()
                )
            except Exception as e:
                log.warning("A/B compare build failed: %s", e)

            return context

        # Fallback: single-run mode
        single_ctx = await self._emit_single_run(
            run_id=run_id_root,
            scorables=scorables,
            context=context,
            target_vec=target_vec,
        )
        context.update(single_ctx)

        return context

    # ---------- CORE EMIT FOR ONE RUN ----------
    async def _emit_single_run(
        self,
        *,
        run_id: str,
        scorables: List[Dict[str, Any]],
        context: Dict[str, Any],
        target_vec: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        out_dir = Path(self.vpm_out) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        self._init_snapshots(out_dir)

        if not scorables:
            # minimal manifest
            manifest = NexusRunManifest(
                run_id=run_id,
                created_utc=time.time(),
                extras={"goal": context.get("goal"), "source": context.get("source"), "count_scorables": 0},
            )
            manifest.save(out_dir)

            # empty run_metrics + frames so downstream GAP/renderer don't choke
            with (out_dir / "run_metrics.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "nodes": 0, "edges": 0, "avg_degree": 0.0, "mean_edge_weight": 0.0,
                    "connected_components": 0, "largest_component": 0,
                    "clustering_coeff": 0.0, "mutual_knn_frac": 0.0,
                    "spatial": {"mean_edge_len": 0.0, "p10": 0.0, "p90": 0.0},
                    "goal_alignment": {"mean": 0.0, "median": 0.0, "p90": 0.0, "count": 0},
                }, f, indent=2)
            with (out_dir / "frames.json").open("w", encoding="utf-8") as f:
                json.dump([], f)

            log.warning("NexusInlineAgent: run %s had 0 scorables; wrote empty artifacts.", run_id)
            return {
                "nexus_graph_json": (out_dir / "graph.json").as_posix(),  # may not exist; fine
                "nexus_graph_html": (out_dir / "graph.html").as_posix(),  # may not exist; fine
                "nexus_manifest_path": (out_dir / "manifest.json").as_posix(),
                "nexus_run_dir": out_dir.as_posix(),
            }

        dims_for_vpm = self.dimensions

        # Start a new ZeroModel/VPM run
        with _stage(log, "Nexus.vpm.start_run", run=run_id):
            self.vpmw.start_run(
                run_id, metrics=dims_for_vpm, out_dir=str(out_dir)
            )

        manifest = NexusRunManifest(
            run_id=run_id,
            created_utc=time.time(),
            extras={
                "goal": context.get("goal"),
                "source": context.get("source"),
                "count_scorables": len(scorables),
            },
        )

        # optionally reorder to "think" across time
        strategy = (self.rollout_strategy or "none").lower()
        if strategy == "consensus-walk":
            goal_vec = target_vec or []
            with _stage(log, "Nexus.consensus_walk_order", n=len(scorables)):
                order = _consensus_walk_order(
                    scorables,
                    goal_vec=goal_vec,
                    alpha=float(
                        self.cfg.get("ab_compare", {}).get("alpha", 0.7)
                    ),
                )
            scorables = [scorables[i] for i in order]

        # Per-item metrics + VPM tiles
        task_items = f"nexus:items:{run_id}"
        self.pstart(
            task=task_items, total=len(scorables), meta={"phase": "items"}
        )
        with _stage(log, "Nexus.items", count=len(scorables), run=run_id):
            for idx, s in enumerate(scorables):
                goal_for_item = s.get("goal_ref") or context.get("goal")
                merged_context = {**context, "goal": goal_for_item}
                sc = Scorable.from_dict(s)

                with _stage(log, "Nexus.score_and_append"):
                    mx = await self.mxw.score_and_append(
                        self.zm, sc, context=merged_context, run_id=run_id
                    )

                item_name = sc.id or f"item-{idx:04d}"
                item_dir = out_dir / item_name
                item_dir.mkdir(parents=True, exist_ok=True)

                with _stage(log, "Nexus.vpm.run_item", item=item_name):
                    vpm_rec = await self.vpmw.run_item(
                        run_id,
                        sc,
                        out_dir=str(item_dir),
                        dims_for_score=dims_for_vpm,
                        rollout_steps=int(self.rollout_steps),
                        rollout_strategy=self.rollout_strategy,
                        save_channels=False,
                        name_hint=item_name,
                    )

                embeddings = dict(s.get("embeddings") or {})
                if "global" in embeddings:
                    embeddings["global"] = as_list_floats(embeddings["global"])

                domains = list(s.get("domains") or [])
                ner = list(s.get("ner") or [])
                tile = _find_tile_path(item_dir)
                item = ManifestItem(
                    item_id=item_name,
                    scorable_id=sc.id or item_name,
                    scorable_type=str(sc.target_type),
                    turn_index=s.get("turn_index"),
                    chat_id=s.get("chat_id"),
                    domains=domains,
                    ner=ner,
                    near_identity=dict(s.get("near_identity") or {}),
                    metrics_columns=list(mx["columns"]),
                    metrics_values=[float(v) for v in mx["values"]],
                    metrics_vector={
                        k: float(v) for k, v in mx["vector"].items()
                    },
                    embeddings={
                        k: as_list_floats(v) for k, v in embeddings.items()
                    },
                    vpm_png=str(tile.as_posix()) if tile else None,
                    rollout=vpm_rec or {},
                )

                with (item_dir / "metrics.json").open(
                    "w", encoding="utf-8"
                ) as f:
                    f.write(
                        dumps_safe(
                            {
                                "item": item.item_id,
                                "scores": mx.get("scores", {}),
                                "metrics_columns": item.metrics_columns,
                                "metrics_values": item.metrics_values,
                                "vector": item.metrics_vector,
                                "embeddings": item.embeddings,
                                "domains": item.domains,
                                "entities": item.ner,
                                "rollout": item.rollout,
                            },
                            indent=2,
                        )
                    )

                manifest.append(item)

                # lightweight step snapshot
                self._append_snapshot(
                    step=idx + 1,
                    total=len(scorables),
                    item_id=item.item_id,
                    meta={
                        "goal_preview": (context.get("goal") or {}).get("goal_text", "")[:160],
                        "has_tile": bool(tile),
                    },
                )

                self.ptick(task=task_items, done=idx + 1, total=len(scorables))
        self.pdone(task=task_items)

        # Finalize ZeroModel run
        with _stage(log, "Nexus.vpm.finalize", run=run_id):
            await self.vpmw.finalize(run_id, out_dir=str(out_dir))

        # Persist manifest
        with _stage(log, "Nexus.manifest.save"):
            manifest.save(out_dir)

        # Graph build
        with _stage(log, "Nexus.graph.nodes"):
            nodes = build_nodes_from_manifest(manifest)

        items_list = [mi.to_dict() for mi in manifest.items]

        with _stage(log, "Nexus.graph.edges", nodes=len(nodes)):
            # --- Small-N guard to avoid saturated cliques ---
            n = len(nodes)
            k_cfg = int(self.cfg.get("indexer", {}).get("knn", {}).get("k", 12))
            edge_thr_cfg = float(
                self.cfg.get("indexer", {}).get("knn", {}).get("edge_threshold", 0.35)
            )

            # Clamp K to something sensible for tiny graphs, and bump threshold a bit.
            # Example: with n=10 → k ≈ 3, and min threshold 0.50 to avoid full connectivity
            k_clamped = max(2, min(k_cfg, max(3, n // 3))) if n > 0 else 2
            edge_thr = edge_thr_cfg
            if n <= 20:
                edge_thr = max(edge_thr_cfg, 0.50)

            # Optional: expose overrides via config
            k_clamped = int(self.cfg.get("indexer", {}).get("knn", {}).get("k_effective", k_clamped))
            edge_thr = float(self.cfg.get("indexer", {}).get("knn", {}).get("edge_threshold_effective", edge_thr))

            self.logger.log("NexusKNNParams", {
                "n_nodes": n,
                "k_cfg": k_cfg,
                "k_effective": k_clamped,
                "edge_threshold_cfg": edge_thr_cfg,
                "edge_threshold_effective": edge_thr,
            })

            edges = build_edges_enhanced(
                run_id=run_id,
                nodes=nodes,
                items=items_list,
                knn_k=k_clamped,
                add_temporal=bool(self.cfg.get("pathfinder", {}).get("backtrack", True)),
                sim_threshold=edge_thr,
            )

        with _stage(
            log, "Nexus.graph.layout", nodes=len(nodes), edges=len(edges)
        ):
            positions = compute_positions(nodes, edges)

        with _stage(log, "Nexus.graph.export.json"):
            export_graph_json(
                Path(out_dir) / "graph.json", nodes, edges, positions
            )

        run_metrics = compute_run_metrics(
            manifest=manifest,
            nodes=nodes,
            edges=edges,
            positions=positions,
            target_vec=target_vec or [],
            params={
                "knn_k_effective": int(k_clamped),
                "edge_threshold_effective": float(edge_thr),
                "knn_k_cfg": int(k_cfg),
                "edge_threshold_cfg": float(edge_thr_cfg),
            }
        )
        with (out_dir / "run_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(run_metrics, f, indent=2)

        # HTML exports (optional toggles)
        try:
            task = f"nexus:exports:{run_id}"
            total_exports = int(self.enable_html) + int(self.enable_rich)
            if total_exports:
                self.pstart(
                    task=task, total=total_exports, meta={"phase": "exports"}
                )
                done = 0
                if self.enable_html:
                    with _stage(log, "Nexus.graph.export.pyvis"):
                        export_pyvis_html(
                            output_path=(out_dir / "graph.html").as_posix(),
                            nodes=nodes,
                            edges=edges,
                            title=f"Nexus Graph — {run_id}",
                        )
                    done += 1
                    self.ptick(task=task, done=done, total=total_exports)
                if self.enable_rich:
                    with _stage(log, "Nexus.graph.export.pyvis_rich"):
                        export_pyvis_html_rich(
                            output_path=(
                                out_dir / "graph_rich.html"
                            ).as_posix(),
                            nodes=nodes,
                            edges=edges,
                            positions=positions,
                            title=f"Nexus Graph — {run_id}",
                        )
                    done += 1
                    self.ptick(task=task, done=done, total=total_exports)
                self.pdone(task=task)
                print(
                    f"Exported PyVis HTML to {(out_dir / 'graph.html').as_posix()}"
                    + (", and graph_rich.html" if self.enable_rich else "")
                )
        except Exception as e:
            log.warning("PyVis export failed: %s", e)

        # Build timeline frames (plus optional filmstrip GIF from VPM tiles)
        try:
            if self.enable_filmstrip:
                task = f"nexus:frames:{run_id}"
                self.pstart(task=task, total=1, meta={"phase": "frames"})
                with _stage(log, "Nexus.frames+gif", run=run_id):
                    await _write_frames_and_gif(
                        out_dir=out_dir,
                        manifest=manifest,
                        nodes=nodes,
                        edges=edges,
                        positions=positions,
                        target_vec=target_vec,
                    )
                self.ptick(task=task, done=1, total=1)
                self.pdone(task=task)
        except Exception as e:
            log.warning("timeline frames build failed: %s", e)

        # final snapshot (post-graph)
        self._append_snapshot(
            step="final",
            total=len(manifest.items),
            item_id=None,
            meta={
                "nodes": len(nodes),
                "edges": len(edges),
                "run_dir": out_dir.as_posix(),
            },
        )

        run_ctx = {
            "nexus_graph_json": (out_dir / "graph.json").as_posix(),
            "nexus_graph_html": (out_dir / "graph.html").as_posix(),
            "nexus_manifest_path": (out_dir / "manifest.json").as_posix(),
            "nexus_run_dir": out_dir.as_posix(),
        }
        print(
            f"NexusInlineAgent completed run {run_id}, output in \n{out_dir.as_posix()}"
        )
        return run_ctx

    # ---------- HELPERS ----------
    async def _ensure_embeddings_global(
        self, scorables: List[Dict[str, Any]], progress=None
    ) -> None:
        """
        Ensure each scorable has embeddings.global; compute from text when missing.
        Coerce any present vector to List[float].
        """
        n = len(scorables)
        for i, s in enumerate(scorables, 1):
            emb_map = s.get("embeddings") or {}
            g = emb_map.get("global")
            if has_vec(g):
                s.setdefault("embeddings", {})["global"] = as_list_floats(g)
                if progress:
                    progress(i, n)
                continue

            txt = s.get("text") or s.get("title") or s.get("payload") or ""
            vec = as_list_floats(
                self.memory.embedding.get_or_create(str(txt)[:4096])
            )
            s.setdefault("embeddings", {})["global"] = vec
            if progress:
                progress(i, n)

    def _plan_ab_from_scorables(
        self,
        scorables: List[Dict[str, Any]],
        goal_vec: List[float],
        *,
        top_k: int,
        seed: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Baseline: random sample of size top_k (seeded).
        Targeted: top_k by cosine to goal_vec using embeddings.global (fallbacks apply).
        Returns (baseline_ids, targeted_ids).
        """
        n = len(scorables)
        k = min(top_k, n)
        rng = random.Random(seed)

        # Baseline random
        base_ids = list(range(n))
        rng.shuffle(base_ids)
        base_ids = base_ids[:k]

        # Targeted by cosine
        def vec_of(i: int) -> List[float]:
            s = scorables[i]
            em = (s.get("embeddings") or {}).get("global")
            v = as_list_floats(em)
            if v:
                return v
            mv = s.get("metrics_vector")
            if isinstance(mv, dict) and mv:
                try:
                    return [float(vv) for vv in mv.values()]
                except Exception:
                    pass
            txt = s.get("text") or s.get("title") or s.get("payload") or ""
            return as_list_floats(
                self.memory.embedding.get_or_create(str(txt)[:4096])
            )

        scores = [(i, cos_safe(vec_of(i), goal_vec)) for i in range(n)]
        scores.sort(key=lambda t: -t[1])
        tgt_ids = [i for i, _ in scores[:k]]

        return base_ids, tgt_ids

    # ---------- snapshot helpers ----------
    def _init_snapshots(self, run_dir: Path) -> None:
        try:
            self._snap_path = run_dir / "snapshots.jsonl"
            if self._snap_path.exists():
                self._snap_path.unlink()
        except Exception:
            self._snap_path = None

    def _append_snapshot(self, *, step: Any, total: int, item_id: Optional[str], meta: Dict[str, Any]) -> None:
        if not self._snap_path:
            return
        rec = {
            "ts": int(time.time()),
            "step": step,
            "total": int(total),
            "item_id": item_id,
            "meta": meta or {},
        }
        try:
            with self._snap_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass


def _annotate_with_sim(tile: Image.Image, sim: float | None) -> Image.Image:
    if sim is None:
        return tile
    s = max(0.0, min(1.0, float(sim)))
    w, h = tile.size
    bar_h = 10
    out = Image.new("RGB", (w, h + bar_h), "black")
    out.paste(tile, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, h, int(s * w), h + bar_h], fill=(0, 255, 0))
    return out


# --------- Timeline + Filmstrip builder ----------
async def _write_frames_and_gif(
    out_dir: Path | str,
    manifest: NexusRunManifest,
    nodes: Dict[str, Any],
    edges: List[Any],
    positions: Dict[str, Any],
    target_vec: Optional[List[float]] = None,
) -> Optional[str]:
    """
    Build frames.json with progressive nodes/edges and per-frame metrics,
    and render an animated filmstrip GIF from item VPMs if present.
    """
    out_dir = Path(out_dir)

    # Order map
    item_ids = [mi.item_id for mi in manifest.items]
    order = {iid: i for i, iid in enumerate(item_ids)}
    LAST = max(len(item_ids) - 1, 0)

    def _idx(nid: str) -> int:
        return order.get(nid, LAST)

    # Nodes → cytoscape format (preserve preset positions)
    cy_all_nodes = []
    for nid, n in nodes.items():
        x, y = positions.get(nid, (None, None))
        cy_all_nodes.append(
            {
                "data": {
                    "id": nid,
                    "label": getattr(n, "title", None)
                    or getattr(n, "text", "")[:80]
                    or nid,
                    "type": getattr(n, "target_type", "node"),
                    "weight": float(getattr(n, "weight", 0.0) or 0.0),
                },
                "position": None
                if (x is None or y is None)
                else {"x": x, "y": y},
            }
        )

    # Edges by step
    edge_buckets: Dict[int, List[Any]] = {}
    for e in edges:
        s = getattr(e, "src", None) or getattr(e, "source", None)
        t = getattr(e, "dst", None) or getattr(e, "target", None)
        step = max(_idx(str(s)), _idx(str(t)))
        edge_buckets.setdefault(step, []).append(e)

    def _cy_edges(es: List[Any]) -> List[Dict[str, Any]]:
        out = []
        for e in es:
            src = getattr(e, "src", None) or getattr(e, "source", None)
            dst = getattr(e, "dst", None) or getattr(e, "target", None)
            ety = getattr(e, "type", "edge")
            w = float(getattr(e, "weight", 0.0) or 0.0)
            out.append(
                {
                    "data": {
                        "id": f"{src}->{dst}",
                        "source": src,
                        "target": dst,
                        "type": ety,
                        "weight": w,
                    }
                }
            )
        return out

    # Optional similarity trace vs target_vec (use embeddings.global or metrics_vector)
    sims: List[Optional[float]] = []
    for mi in manifest.items:
        v = (mi.embeddings or {}).get("global")
        vec = (
            as_list_floats(v)
            if has_vec(v)
            else [float(x) for x in (mi.metrics_vector or {}).values()]
            if (mi.metrics_vector or {})
            else []
        )
        sims.append(cos_safe(vec, target_vec) if target_vec else None)

    # Build frames
    frames: List[Dict[str, Any]] = []
    used_edges: List[Any] = []
    for k in range(len(item_ids)):
        used_edges.extend(edge_buckets.get(k, []))
        f_nodes = [n for n in cy_all_nodes if _idx(n["data"]["id"]) <= k]
        f_edges = _cy_edges(used_edges)
        frames.append(
            {
                "nodes": f_nodes,
                "edges": f_edges,
                "metrics": {
                    "step": k + 1,
                    "total_steps": len(item_ids),
                    "new_edges": len(edge_buckets.get(k, [])),
                    "new_nodes": sum(
                        1 for n in f_nodes if _idx(n["data"]["id"]) == k
                    ),
                    "sim": sims[k] if k < len(sims) else None,
                },
            }
        )

    # Persist frames.json
    with (out_dir / "frames.json").open("w", encoding="utf-8") as f:
        json.dump(frames, f, ensure_ascii=False, indent=2)

    # Optional: build animated filmstrip from per-item VPMs
    frames_png = []
    for mi, sim in zip(manifest.items, sims):
        tile_path = mi.vpm_png
        tile_img = None

        if tile_path and Path(tile_path).exists():
            tile_img = Image.open(tile_path).convert("RGB")
        else:
            # try to locate something under this item's dir
            item_dir = Path(out_dir) / mi.item_id
            # 1) look for any vpm*.png
            cand = next((p for p in item_dir.rglob("vpm*.png") if p.is_file()), None)
            if cand:
                tile_img = Image.open(cand).convert("RGB")
            else:
                # 2) fallback to first frame of any per-item filmstrip.gif
                gif = next((p for p in item_dir.rglob("filmstrip.gif") if p.is_file()), None)
                if gif:
                    tile_img = _first_frame_from_gif(gif)

        if tile_img:
            tile_img = tile_img.resize((256, 256))
            tile_img = _annotate_with_sim(tile_img, sim)
            frames_png.append(np.array(tile_img))

    if frames_png:
        gif_path = out_dir / "filmstrip.gif"
        imageio.mimsave(gif_path, frames_png, duration=0.35)
        return str(gif_path)
    return None
