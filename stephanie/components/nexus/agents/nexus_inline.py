# stephanie/components/nexus/agents/nexus_inline.py
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
# add to imports at the top
from stephanie.components.nexus.graph.builder import (
    build_edges_enhanced, build_nodes_from_manifest)
from stephanie.components.nexus.graph.exporters import (export_graph_json,
                                                        export_pyvis_html)
from stephanie.components.nexus.graph.layout import compute_positions
from stephanie.components.nexus.graph.timeline import \
    build_frames_from_manifest
from stephanie.components.nexus.manifest import ManifestItem, NexusRunManifest
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.scoring_service import ScoringService
from stephanie.services.workers.nexus_workers import (NexusMetricsWorkerInline,
                                                      NexusVPMWorkerInline)
from stephanie.services.zeromodel_service import ZeroModelService
from stephanie.utils.json_sanitize import dumps_safe  # small helper; see below

log = logging.getLogger(__name__)

class NexusInlineAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.zm: ZeroModelService = self.container.get("zeromodel")
        self.scoring: ScoringService = self.container.get("scoring")
        self.vpmw = NexusVPMWorkerInline(self.zm, logger=logger)
        self.mxw  = NexusMetricsWorkerInline(
            scoring=self.scoring,
            scorers=["sicql", "hrm", "tiny"],
            dimensions=["alignment","clarity","relevance","coverage","faithfulness"],
            persist=False
        )
        self.vpm_out = self.cfg.get("vpm_out", "./runs/nexus_vpm/")
        self.rollout_steps = int(self.cfg.get("rollout_steps", 0))
        self.rollout_strategy = self.cfg.get("rollout_strategy", "none")
        self.target_type = self.cfg.get("target_type", ScorableType.CONVERSATION_TURN)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.zm.initialize()
        dims_for_vpm = ["clarity","coherence","complexity","alignment","coverage"]
        run_id = context.get(PIPELINE_RUN_ID)
        out_dir = Path(self.vpm_out) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        self.vpmw.start_run(run_id, metrics=dims_for_vpm, out_dir=str(out_dir))

        scorables = context.get("scorables", [])
        manifest = NexusRunManifest(run_id=run_id, created_utc=time.time(), extras={
            "goal": context.get("goal"),
            "source": context.get("source"),
            "count_scorables": len(scorables),
        })

        for idx, s in enumerate(scorables):
            goal = s.get("goal_ref") or context.get("goal")
            merged_context = {**context, "goal": goal}
            sc = Scorable.from_dict(s)

            # A) dense text metrics row (vector + columns)
            mx = await self.mxw.score_and_append(self.zm, sc, context=merged_context, run_id=run_id)
            # mx has: model_alias, columns, values, vector, scores (see MetricsWorkerInline)
            #       -> we’ll put vector into manifest item

            # B) VPM + (optional) rollout rows
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

            # C) embeddings/domains/entities if present on the scorable
            # expecting: s["embeddings"] = {"global": [...], "goal":[...]} etc.
            embeddings = dict(s.get("embeddings") or {})
            domains    = list(s.get("domains") or [])
            ner   = list(s.get("ner") or [])

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
                embeddings={k: list(map(float, v)) for k, v in embeddings.items()},
                vpm_png=str((item_dir / "vpm.png").as_posix()) if (item_dir / "vpm.png").exists() else None,
                rollout=vpm_rec or {},
            )

            # optional: dump per-item JSON for quick inspection
            with (item_dir / "metrics.json").open("w", encoding="utf-8") as f:
                f.write(dumps_safe({
                    "item": item.item_id,
                    "scores": mx.get("scores", {}),
                    "metrics_columns": item.metrics_columns,
                    "metrics_values": item.metrics_values,
                    "vector": item.metrics_vector,
                    "embeddings": item.embeddings,
                    "domains": item.domains,
                    "entities": item.ner,
                    "rollout": item.rollout,
                }, indent=2))

            manifest.append(item)

        # finalize ZeroModel timeline
        await self.vpmw.finalize(run_id, out_dir=str(out_dir))

        # write manifest.json
        manifest.save(out_dir)

                # -----------------------------
        # Build & persist the graph
        # -----------------------------
        # 1) Build nodes from the manifest
        nodes = build_nodes_from_manifest(manifest)  # {id -> NexusNode}

        # 2) Build edges (KNN + temporal) from items (pass the raw list/dicts)
        items_list = [mi.to_dict() for mi in manifest.items]   # or manifest.as_dict()["items"]
        edges = build_edges_enhanced(
            run_id=run_id,
            nodes=nodes,
            items=items_list,
            knn_k=int(self.cfg.get("indexer", {}).get("knn", {}).get("k", 12)),
            add_temporal=bool(self.cfg.get("pathfinder", {}).get("backtrack", True)),  # or True if you always want temporal
            sim_threshold=float(self.cfg.get("indexer", {}).get("knn", {}).get("edge_threshold", 0.35)),
        )

        # 3) Save graph.json for the Cytoscape UI
        positions = compute_positions(nodes, edges)
        export_graph_json(Path(out_dir) / "graph.json", nodes, edges, positions)

        # 4) Optional: quick PyVis HTML (nice for ad-hoc viewing)
        try:
            export_pyvis_html(
                output_path=(out_dir / "graph.html").as_posix(),
                nodes=nodes,
                edges=edges,
                title=f"Nexus Graph — {run_id}"
            )
        except Exception as e:
            log.warning("PyVis export failed: %s", e)

        # 3b) Build a timeline frames.json (progressive reveal by item order)
        try:
            frames = build_frames_from_manifest(
                manifest_items=manifest.items,
                nodes=nodes,
                edges=edges,
                positions=positions,
                progress=lambda d, t: log.info("timeline %d/%d", d, t),
                tqdm_progress=False,          # set True if you want a console bar
                # stream_to_path=(out_dir / "frames.json").as_posix(),  # fastest option
            )
            if frames:  # when not streaming
                with (out_dir / "frames.json").open("w", encoding="utf-8") as f:
                    json.dump(frames, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("timeline frames build failed: %s", e)

        # 5) Hand paths to the router / pipeline
        context["nexus_graph_json"] = (out_dir / "graph.json").as_posix()
        context["nexus_graph_html"] = (out_dir / "graph.html").as_posix()


        # echo paths for the pipeline
        context["nexus_manifest_path"] = str((out_dir / "manifest.json").as_posix())
        context["nexus_run_dir"] = str(out_dir.as_posix())

        print(f"NexusInlineAgent completed run {run_id}, output in \n{str(out_dir.as_posix())}")
        return context
