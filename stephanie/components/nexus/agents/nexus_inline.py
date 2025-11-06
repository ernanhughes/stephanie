# stephanie/components/nexus/agents/nexus_inline.py
from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.nexus.graph.builder import (
    build_edges_enhanced,
    build_nodes_from_manifest,
)
from stephanie.components.nexus.graph.exporters import (
    export_graph_json,
    export_pyvis_html,
    export_pyvis_html_rich
)
from stephanie.components.nexus.graph.layout import compute_positions
from stephanie.components.nexus.manifest import ManifestItem, NexusRunManifest
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.scoring_service import ScoringService
from stephanie.services.workers.nexus_workers import (
    NexusMetricsWorkerInline,
    NexusVPMWorkerInline,
)
from stephanie.services.zeromodel_service import ZeroModelService
from stephanie.utils.json_sanitize import dumps_safe
from stephanie.tools.embed_utils import as_list_floats, has_vec, cos_safe

log = logging.getLogger(__name__)

class NexusInlineAgent(BaseAgent):
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
        self.dimensions = cfg.get("dimensions", ["alignment", "clarity", "relevance", "coverage", "faithfulness"])

        self.vpmw = NexusVPMWorkerInline(self.zm, logger=logger)
        self.mxw = NexusMetricsWorkerInline(
            scoring=self.scoring,
            scorers=self.scorers,
            dimensions=self.dimensions,
            persist=False,
        )

        self.vpm_out = self.cfg.get("vpm_out", "./runs/nexus_vpm/")
        self.rollout_steps = int(self.cfg.get("rollout_steps", 0))
        self.rollout_strategy = self.cfg.get("rollout_strategy", "none")
        self.target_type = self.cfg.get(
            "target_type", ScorableType.CONVERSATION_TURN
        )
        self.use_ab = bool(self.cfg.get("ab_compare", {}).get("enabled", True))

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.zm.initialize()

        run_id_root = context.get(PIPELINE_RUN_ID)
        scorables: List[Dict[str, Any]] = list(context.get("scorables", []))

        # Ensure each scorable has embeddings.global (computed from text if missing)
        await self._ensure_embeddings_global(scorables)

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

            baseline_ids, targeted_ids = self._plan_ab_from_scorables(
                scorables, target_vec, top_k=top_k, seed=seed
            )
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
        dims_for_vpm = self.dimensions
        out_dir = Path(self.vpm_out) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Start a new ZeroModel/VPM run
        self.vpmw.start_run(run_id, metrics=dims_for_vpm, out_dir=str(out_dir))

        manifest = NexusRunManifest(
            run_id=run_id,
            created_utc=time.time(),
            extras={
                "goal": context.get("goal"),
                "source": context.get("source"),
                "count_scorables": len(scorables),
            },
        )

        # Per-item metrics + VPM tiles
        for idx, s in enumerate(scorables):
            goal_for_item = s.get("goal_ref") or context.get("goal")
            merged_context = {**context, "goal": goal_for_item}
            sc = Scorable.from_dict(s)

            mx = await self.mxw.score_and_append(
                self.zm, sc, context=merged_context, run_id=run_id
            )

            item_name = sc.id or f"item-{idx:04d}"
            item_dir = out_dir / item_name
            item_dir.mkdir(parents=True, exist_ok=True)

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
            # Normalize embeddings.global to List[float] if present
            if "global" in embeddings:
                embeddings["global"] = as_list_floats(embeddings["global"])

            domains = list(s.get("domains") or [])
            ner = list(s.get("ner") or [])

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
                metrics_vector={k: float(v) for k, v in mx["vector"].items()},
                embeddings={
                    k: as_list_floats(v) for k, v in embeddings.items()
                },
                vpm_png=str((item_dir / "vpm.png").as_posix())
                if (item_dir / "vpm.png").exists()
                else None,
                rollout=vpm_rec or {},
            )

            with (item_dir / "metrics.json").open("w", encoding="utf-8") as f:
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

        # Finalize ZeroModel run
        await self.vpmw.finalize(run_id, out_dir=str(out_dir))

        # Persist manifest
        manifest.save(out_dir)

        # Graph build
        nodes = build_nodes_from_manifest(manifest)
        items_list = [mi.to_dict() for mi in manifest.items]
        edges = build_edges_enhanced(
            run_id=run_id,
            nodes=nodes,
            items=items_list,
            knn_k=int(self.cfg.get("indexer", {}).get("knn", {}).get("k", 12)),
            add_temporal=bool(
                self.cfg.get("pathfinder", {}).get("backtrack", True)
            ),
            sim_threshold=float(
                self.cfg.get("indexer", {})
                .get("knn", {})
                .get("edge_threshold", 0.35)
            ),
        )

        positions = compute_positions(nodes, edges)
        export_graph_json(
            Path(out_dir) / "graph.json", nodes, edges, positions
        )

        try:
            export_pyvis_html_rich(
                output_path=(out_dir / "graph.html").as_posix(),
                nodes=nodes,
                edges=edges,
                positions=positions,
                title=f"Nexus Graph — {run_id}",
            )
            print(f"Exported rich PyVis HTML to {(out_dir / 'graph.html').as_posix()}")
        except Exception as e:
            log.warning("PyVis export failed: %s", e)

        # Build timeline frames (plus optional filmstrip GIF from VPM tiles)
        try:
            await _write_frames_and_gif(
                out_dir=out_dir,
                manifest=manifest,
                nodes=nodes,
                edges=edges,
                positions=positions,
                target_vec=target_vec,
            )
        except Exception as e:
            log.warning("timeline frames build failed: %s", e)

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
        self, scorables: List[Dict[str, Any]]
    ) -> None:
        """
        Ensure each scorable has embeddings.global; compute from text when missing.
        Coerce any present vector to List[float].
        """
        for s in scorables:
            emb_map = s.get("embeddings") or {}
            g = emb_map.get("global")
            if has_vec(g):
                # normalize in place
                s.setdefault("embeddings", {})["global"] = as_list_floats(g)
                continue

            txt = s.get("text") or s.get("title") or s.get("payload") or ""
            vec = as_list_floats(
                self.memory.embedding.get_or_create(str(txt)[:4096])
            )
            s.setdefault("embeddings", {})["global"] = vec

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


def _annotate_with_sim(tile: Image.Image, sim: float|None) -> Image.Image:
    if sim is None: 
        return tile
    # clamp to [0,1]
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
        if mi.vpm_png and Path(mi.vpm_png).exists():
            tile = Image.open(mi.vpm_png).convert("RGB").resize((256, 256))
            tile = _annotate_with_sim(tile, sim)  # <-- add the bar
            frames_png.append(np.array(tile))
    if frames_png:
        gif_path = out_dir / "filmstrip.gif"
        imageio.mimsave(
            gif_path, [np.array(im) for im in frames_png], duration=0.35
        )
        return str(gif_path)
    return None
