# stephanie/components/gap/processors/analysis.py
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import time
import json, os
from collections import Counter

from stephanie.components.gap.models import GapConfig
from stephanie.components.gap.processors.topology import (TopologyConfig,
                                                          TopologyProcessor)
from stephanie.components.gap.processors.visuals import render_scm_images
from stephanie.utils.progress_mixin import ProgressMixin

_logger = logging.getLogger(__name__)




class AnalysisProcessor(ProgressMixin):
    """
    Orchestrates analytical computations for GAP system.
    Uses ProgressMixin for structured progress without passing callbacks.
    """

    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger
        self.topology_processor: TopologyProcessor | None = None
        self._init_progress(container, logger)  # <-- ProgressService hookup
        _logger.debug(f"AnalysisProcessor initialized with config: {config}")

    async def execute_analysis(
        self,
        scoring_results: Dict[str, Any],
        run_id: str,
        *,
        manifest: Any | None = None,
    ) -> Dict[str, Any]:
        task = f"analysis:{run_id}"
        total_stages = 6  # frontier, delta, intensity, phos, scm_visuals, topology
        self.pstart(task, total=total_stages, meta={"run_id": run_id})

        # --- NEW: small helpers to keep manifest logging DRY -------------------
        def _m_start(stage_name: str, extra: dict | None = None):
            if not manifest: 
                return
            manifest.stage_start(
                stage_name,
                total=None,
                run_id=run_id,
                status="running",
                started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **(extra or {}),
            )

        def _m_end(stage_name: str, status: str = "ok", artifacts: dict | None = None, extra: dict | None = None):
            if not manifest:
                return
            manifest.stage_end(
                stage_name,
                status=status,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                artifacts=(artifacts or {}),
                **(extra or {}),
            )

        def _pick_artifacts(d: dict) -> dict:
            """Heuristic: keep paths to files/images that are useful for the manifest."""
            out = {}
            for k, v in (d or {}).items():
                if isinstance(v, str) and any(v.lower().endswith(ext) for ext in (
                    ".png", ".jpg", ".jpeg", ".webp", ".gif",
                    ".json", ".npy", ".npz", ".parquet", ".csv", ".pdf"
                )):
                    out[k] = v
                # lists of paths (e.g., scm images)
                if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                    if any(str(x).lower().endswith((".png",".json",".npy",".parquet",".gif",".jpg",".jpeg",".webp")) for x in v):
                        out[k] = v
                # nested dict with "paths"
                if isinstance(v, dict) and "paths" in v and isinstance(v["paths"], dict):
                    out[f"{k}_paths"] = v["paths"]
            return out
        # ----------------------------------------------------------------------

        _logger.debug(f"Starting analysis pipeline for run {run_id}")
        _logger.debug(f"Scoring results keys: {list(scoring_results.keys())}")

        zm = self.container.get("zeromodel")

        out_metrics_dir = self.config.base_dir / run_id / "metrics"
        out_visuals_dir = self.config.base_dir / run_id / "visuals"
        out_metrics_dir.mkdir(parents=True, exist_ok=True)
        out_visuals_dir.mkdir(parents=True, exist_ok=True)

        # Resolve matrices (prefers native vectors)
        self.pstage(task, "resolve_matrices")
        hrm_matrix, tiny_matrix, hrm_names, tiny_names = self._resolve_mats_and_names(scoring_results)
        self.pstage(task, "resolve_matrices:done", status="ok",
                    shapes={"hrm": getattr(hrm_matrix, "shape", None),
                            "tiny": getattr(tiny_matrix, "shape", None)})

        results: Dict[str, Any] = {}
        done_stages = 0

        # --- Frontier ---
        self.pstage(task, "frontier:start")
        _m_start("analysis.frontier")
        try:
            res = await self._perform_frontier(
                zm, hrm_matrix, tiny_matrix, run_id,
                scoring_results.get("hrm_label", "HRM"),
                scoring_results.get("tiny_label", "Tiny"),
            )
            results["frontier"] = res
            self.pstage(task, "frontier:done", status="ok")
            _m_end("analysis.frontier", status="ok",
                artifacts=_pick_artifacts(res),
                extra={"summary": {k:v for k,v in res.items() if isinstance(v,(int,float,str))}})
        except Exception as e:
            _logger.error(f"Frontier analysis failed: {e}", exc_info=True)
            results["frontier"] = {"error": str(e)}
            self.pstage(task, "frontier:done", status="error", error=str(e))
            _m_end("analysis.frontier", status="error", artifacts={"error": str(e)})
        done_stages += 1
        self.ptick(task, done=done_stages, total=total_stages)

        # --- Delta analysis ---
        self.pstage(task, "delta:start")
        _m_start("analysis.delta")
        try:
            res = await self._perform_delta_analysis(zm, hrm_matrix, tiny_matrix, hrm_names, tiny_names, run_id)
            results["delta_analysis"] = res
            self.pstage(task, "delta:done", status="ok")
            _m_end("analysis.delta", status="ok",
                artifacts=_pick_artifacts(res),
                extra={"metrics": {k:v for k,v in res.get("metrics", {}).items() if isinstance(v,(int,float))}})
        except Exception as e:
            _logger.error(f"Delta analysis failed: {e}", exc_info=True)
            results["delta_analysis"] = {"error": str(e)}
            self.pstage(task, "delta:done", status="error", error=str(e))
            _m_end("analysis.delta", status="error", artifacts={"error": str(e)})
        done_stages += 1
        self.ptick(task, done=done_stages, total=total_stages)

        # --- Intensity report ---
        self.pstage(task, "intensity:start")
        _m_start("analysis.intensity")
        try:
            res = await self._generate_intensity_report(zm, hrm_matrix, tiny_matrix, hrm_names, tiny_names, run_id)
            results["intensity"] = res
            self.pstage(task, "intensity:done", status="ok")
            _m_end("analysis.intensity", status="ok", artifacts=_pick_artifacts(res))
        except Exception as e:
            _logger.error(f"Intensity report generation failed: {e}", exc_info=True)
            results["intensity"] = {"error": str(e)}
            self.pstage(task, "intensity:done", status="error", error=str(e))
            _m_end("analysis.intensity", status="error", artifacts={"error": str(e)})
        done_stages += 1
        self.ptick(task, done=done_stages, total=total_stages)

        # --- PHOS analysis ---
        self.pstage(task, "phos:start")
        _m_start("analysis.phos")
        try:
            res = await self._perform_phos_analysis(run_id)
            results["phos"] = res
            self.pstage(task, "phos:done", status="ok")
            _m_end("analysis.phos", status="ok", artifacts=_pick_artifacts(res))
        except Exception as e:
            _logger.error(f"PHOS analysis failed: {e}", exc_info=True)
            results["phos"] = {"error": str(e)}
            self.pstage(task, "phos:done", status="error", error=str(e))
            _m_end("analysis.phos", status="error", artifacts={"error": str(e)})
        done_stages += 1
        self.ptick(task, done=done_stages, total=total_stages)

        # --- SCM visuals ---
        self.pstage(task, "scm_visuals:start")
        _m_start("analysis.scm_visuals")
        try:
            hrm_scm = scoring_results.get("hrm_scm_matrix")
            tiny_scm = scoring_results.get("tiny_scm_matrix")
            scm_names = scoring_results.get("scm_names", [])

            if hrm_scm is None or tiny_scm is None or not scm_names:
                storage = self.container.get("gap_storage")
                aligned = storage.base_dir / run_id / "aligned"
                hrm_scm = np.load(aligned / "hrm_scm_matrix.npy")
                tiny_scm = np.load(aligned / "tiny_scm_matrix.npy")
                import json
                with open(aligned / "hrm_scm_metric_names.json", "r", encoding="utf-8") as f:
                    scm_names = json.load(f)

            scm_dir = self.config.base_dir / run_id / "visuals" / "scm"
            pos_label = scoring_results.get("hrm_label", "HRM")
            neg_label = scoring_results.get("tiny_label", "Tiny")
            img_paths = render_scm_images(hrm_scm, tiny_scm, scm_names, scm_dir, pos_label, neg_label)
            results["scm_visuals"] = img_paths
            self.pstage(task, "scm_visuals:done", status="ok", n_imgs=len(img_paths))
            _m_end("analysis.scm_visuals", status="ok", artifacts={"images": img_paths})
        except Exception as e:
            _logger.error(f"SCM visualization failed: {e}", exc_info=True)
            results["scm_visuals"] = {"error": str(e)}
            self.pstage(task, "scm_visuals:done", status="error", error=str(e))
            _m_end("analysis.scm_visuals", status="error", artifacts={"error": str(e)})
        done_stages += 1
        self.ptick(task, done=done_stages, total=total_stages)

        # --- Topology (UMAP + PH) ---
        self.pstage(task, "topology:start")
        _m_start("analysis.topology")
        try:
            self.topology_processor = TopologyProcessor(
                TopologyConfig(
                    use_weighted=True,
                    weights={
                        "reasoning.score01": 1.3,
                        "knowledge.score01": 1.1,
                        "clarity.score01": 1.0,
                        "faithfulness.score01": 1.2,
                        "coverage.score01": 1.0,
                    },
                    umap_n_neighbors=15,
                    umap_min_dist=0.2,
                    dbscan_eps=0.3,
                    dbscan_min_samples=5,
                    max_betti_dim=1,
                ),
                container=self.container,
                logger=self.logger,
            )
            topo_out = await self.topology_processor.run(run_id, base_dir=self.config.base_dir)
            results["topology"] = topo_out
            self.pstage(task, "topology:done", status="ok")
            _m_end("analysis.topology", status="ok",
                artifacts=_pick_artifacts(topo_out),
                extra={"betti": topo_out.get("betti", {})})
        except Exception as e:
            _logger.error(f"Topology analysis failed: {e}", exc_info=True)
            results["topology"] = {"error": str(e)}
            self.pstage(task, "topology:done", status="error", error=str(e))
            _m_end("analysis.topology", status="error", artifacts={"error": str(e)})
        done_stages += 1
        self.ptick(task, done=done_stages, total=total_stages)

        _logger.info(f"Analysis pipeline completed for run {run_id} with {len(results)} components")
        self.pdone(task, extra={"stages": done_stages})

        if manifest:
            results["manifest"] = manifest.to_dict()

        return results

    # ---------- Matrix resolution ----------
    def _resolve_mats_and_names(
        self, scoring_results: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        _logger.debug("Resolving matrices for analysis...")
        H = scoring_results.get("hrm_vectors")
        T = scoring_results.get("tiny_vectors")

        if isinstance(H, np.ndarray) and isinstance(T, np.ndarray):
            if H.shape == T.shape and H.shape[1] > 0:
                return H, T, list(scoring_results.get("hrm_names", [])), list(scoring_results.get("tiny_names", []))
            _logger.warning(f"Native vector shape mismatch: HRM {getattr(H, 'shape', None)} != Tiny {getattr(T, 'shape', None)}")

        Hs = scoring_results.get("hrm_scm_matrix")
        Ts = scoring_results.get("tiny_scm_matrix")
        if not (isinstance(Hs, np.ndarray) and isinstance(Ts, np.ndarray) and Hs.shape == Ts.shape):
            raise ValueError("SCM fallback unavailable or shape-mismatched.")
        names = list(scoring_results.get("scm_names", []))
        return Hs, Ts, names, names

    # ---------- Individual steps ----------
    async def _perform_frontier(
        self,
        zm,
        pos_matrix: np.ndarray,
        neg_matrix: np.ndarray,
        run_id: str,
        pos_label: str,
        neg_label: str,
    ) -> Dict[str, Any]:
        """
        NOTE: order fixed to match call sites: run_id precedes labels.
        """
        _logger.debug(f"Rendering frontier map for matrices: {pos_matrix.shape}")
        if pos_matrix.shape != neg_matrix.shape:
            raise ValueError(f"Frontier: shape mismatch hrm={pos_matrix.shape} tiny={neg_matrix.shape}")

        out_dir = str(self.config.base_dir / run_id / "visuals")
        return zm.render_frontier_map(
            pos_matrix,
            neg_matrix,
            out_dir=out_dir,
            pos_label=pos_label,
            neg_label=neg_label,
            k_latent=20,
        )

    async def _perform_delta_analysis(
        self,
        zm,
        hrm_matrix: np.ndarray,
        tiny_matrix: np.ndarray,
        hrm_names: list[str],
        tiny_names: list[str],
        run_id: str,
    ) -> Dict[str, Any]:
        if hrm_matrix.shape != tiny_matrix.shape:
            raise ValueError(f"Delta: shape mismatch hrm={hrm_matrix.shape} tiny={tiny_matrix.shape}")

        out_dir = str(self.config.base_dir / run_id / "metrics")
        meta = zm.render_intermodel_delta(
            hrm_matrix,
            tiny_matrix,
            names_A=hrm_names,
            names_B=tiny_names,
            output_dir=out_dir,
            pos_label="HRM",
            neg_label="Tiny",
        )

        # quick visual
        try:
            import matplotlib.pyplot as plt
            Dabs = np.abs(hrm_matrix - tiny_matrix)
            plt.figure(figsize=(8, 5))
            plt.imshow(Dabs, cmap="gray", aspect="auto")
            plt.title("|HRM − Tiny| (aligned)")
            plt.axis("off")
            png = self.config.base_dir / run_id / "visuals" / "delta_heat.png"
            plt.savefig(png, dpi=160, bbox_inches="tight")
            plt.close()
            meta["delta_abs_heat"] = str(png)
        except Exception as e:
            _logger.warning(f"Could not generate delta heatmap: {e}")

        return meta

    async def _generate_intensity_report(
        self,
        zm,
        hrm_matrix: np.ndarray,
        tiny_matrix: np.ndarray,
        hrm_names: list[str],
        tiny_names: list[str],
        run_id: str,
    ) -> Dict[str, Any]:
        """
        Generate intensity report identifying high-variance metrics.
        
        Highlights metrics with:
        - Largest absolute differences
        - Highest variance across samples
        - Statistical outliers
        - Critical performance gaps
        """
        _logger.debug("Generating intensity report...")
        
        out_dir = str(self.config.base_dir / run_id / "metrics")
        intensity = zm.build_intensity_report(
            hrm_matrix=hrm_matrix,
            tiny_matrix=tiny_matrix,
            hrm_metric_names=hrm_names,
            tiny_metric_names=tiny_names,
            out_dir=out_dir,
            top_k=25,
        )
        _logger.debug(f"Intensity report generated with {len(intensity)} entries")
        return intensity

    async def _perform_phos_analysis(self, run_id: str) -> Dict[str, Any]:
        from stephanie.zeromodel.vpm_phos import build_compare_guarded

        df_proj = self._prepare_phos_data(run_id)
        if df_proj is None or df_proj.empty:
            _logger.warning(f"No PHOS data available for run {run_id}")
            return {"status": "no_rows_for_df"}

        # 1) Get aliases (manifest → df inference → fallback)
        alias_a, alias_b = self._read_manifest_aliases(run_id)
        if not alias_a or not alias_b:
            ia, ib = self._infer_aliases_from_df(df_proj, self.config.dimensions)
            alias_a = alias_a or ia
            alias_b = alias_b or ib
        if not alias_a or not alias_b:
            # Final fallback to legacy names
            alias_a = alias_a or "hrm"
            alias_b = alias_b or "tiny"

        # 2) Ensure df has columns for those aliases (copy from hrm./tiny. if needed)
        self._ensure_alias_columns(df_proj, alias_a, alias_b, self.config.dimensions)

        # 3) Missing dims check must use aliases
        missing_dims = [
            d for d in self.config.dimensions
            if f"{alias_a}.{d}" not in df_proj.columns or f"{alias_b}.{d}" not in df_proj.columns
        ]
        if missing_dims:
            self.logger.log("PHOSMissingDims", {
                "run_id": run_id,
                "aliases": {"A": alias_a, "B": alias_b},
                "missing_dims": missing_dims
            })

        # 4) Build outputs (pass required keyword-only args)
        out_prefix = str(self.config.base_dir / run_id / "visuals" / "vpm")
        return build_compare_guarded(
            df_proj,
            dimensions=self.config.dimensions,
            out_prefix=out_prefix,
            model_A=alias_a,
            model_B=alias_b,
            tl_fracs=(0.25, 0.16, 0.36, 0.09),
            delta=0.02,
            interleave=self.config.interleave,
            weights=None,
        )

    def _prepare_phos_data(self, run_id: str):
        import pandas as pd
        storage = self.container.get("gap_storage")
        raw_dir = storage.base_dir / run_id / "raw"
        pq = raw_dir / "rows_for_df.parquet"
        csv = raw_dir / "rows_for_df.csv"

        if pq.exists():
            df = pd.read_parquet(pq)
        elif csv.exists():
            df = pd.read_csv(csv)
        else:
            self.logger.log("PHOSRowsMissing", {"run_id": run_id, "raw_dir": str(raw_dir)})
            return pd.DataFrame()

        # keep = ["node_id"] + [c for c in df.columns if isinstance(c, str) and (c.startswith("hrm.") or c.startswith("tiny."))]
        return df

    def _read_manifest_aliases(self, run_id: str) -> tuple[str|None, str|None]:
        """Try to read aliases saved earlier in manifest.json."""
        mpath = os.path.join(str(self.config.base_dir), run_id, "manifest.json")
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                m = json.load(f)
            # Prefer explicit aliases if present; else fall back to models.{A,B}
            aliases = m.get("aliases") or {}
            a = aliases.get("A") or (m.get("models") or {}).get("A")
            b = aliases.get("B") or (m.get("models") or {}).get("B")
            return (str(a) if a else None, str(b) if b else None)
        except Exception:
            return (None, None)

    def _infer_aliases_from_df(self, df_proj, dims: list[str]) -> tuple[str|None, str|None]:
        """Infer aliases from df_proj column prefixes like '<alias>.<dim>'."""
        pref = []
        for c in df_proj.columns:
            if not isinstance(c, str) or c == "node_id" or "." not in c:
                continue
            alias, rest = c.split(".", 1)
            base_dim = rest.split(".", 1)[0]
            if base_dim in dims:
                pref.append(alias)
        if not pref:
            return (None, None)
        counts = Counter(pref).most_common()
        if len(counts) == 1:
            return (counts[0][0], None)
        # Keep stable, friendly ordering: 'hrm' first if present; otherwise by count then name
        aliases = [a for a, _ in counts]
        if "hrm" in aliases:
            a = "hrm"
            aliases.remove("hrm")
            b = aliases[0] if aliases else None
            return (a, b)
        return (aliases[0], aliases[1])

    def _ensure_alias_columns(self, df_proj, alias_a: str, alias_b: str, dims: list[str]):
        """
        If df_proj only has hrm./tiny. but we want custom aliases, copy columns so both exist.
        Non-destructive: keeps originals, adds missing alias columns if needed.
        """
        import pandas as pd
        cols = set(df_proj.columns)
        for dim in dims:
            # A
            want = f"{alias_a}.{dim}"
            if want not in cols:
                if f"hrm.{dim}" in cols:
                    df_proj[want] = pd.to_numeric(df_proj[f"hrm.{dim}"], errors="coerce").fillna(0.0)
                elif f"tiny.{dim}" in cols and alias_a.lower().startswith("tiny"):
                    df_proj[want] = pd.to_numeric(df_proj[f"tiny.{dim}"], errors="coerce").fillna(0.0)
            # B
            want = f"{alias_b}.{dim}"
            if want not in cols:
                if f"tiny.{dim}" in cols:
                    df_proj[want] = pd.to_numeric(df_proj[f"tiny.{dim}"], errors="coerce").fillna(0.0)
                elif f"hrm.{dim}" in cols and alias_b.lower().startswith("hrm"):
                    df_proj[want] = pd.to_numeric(df_proj[f"hrm.{dim}"], errors="coerce").fillna(0.0)
