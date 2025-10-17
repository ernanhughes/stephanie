# stephanie/components/gap/processors/analysis.py
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

from stephanie.components.gap.models import GapConfig
from stephanie.components.gap.processors.topology import TopologyProcessor, TopologyConfig
from stephanie.components.gap.processors.visuals import render_scm_images

_logger = logging.getLogger(__name__)


class AnalysisProcessor:
    """Handles analytical computations: frontier, delta, intensity, PHOS, topology."""

    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger
        self.topology_processor: TopologyProcessor | None = None

    # --- Keep orchestrator API happy ---
    async def execute_analysis(
        self,
        scoring_results: Dict[str, Any],
        run_id: str,
        manifest: Any | None = None,
    ) -> Dict[str, Any]:
        return await self.perform_analysis(scoring_results, run_id, manifest)

    async def perform_analysis(
        self,
        scoring_results: Dict[str, Any],
        run_id: str,
        manifest: Any | None = None,
    ) -> Dict[str, Any]:
        """
        Perform all analytical computations on aligned HRM/Tiny matrices.

        We prefer native alignment if shapes match; otherwise we fall back to SCM matrices.
        """
        zm = self.container.get("zeromodel")

        out_metrics_dir = self.config.base_dir / run_id / "metrics"
        out_visuals_dir = self.config.base_dir / run_id / "visuals"
        out_metrics_dir.mkdir(parents=True, exist_ok=True)
        out_visuals_dir.mkdir(parents=True, exist_ok=True)

        # Resolve matrices/names once, then pass explicitly to helpers
        hrm_matrix, tiny_matrix, hrm_names, tiny_names = self._resolve_mats_and_names(scoring_results)

        results: Dict[str, Any] = {}

        # --- Frontier (aligned difference field) ---
        try:
            results["frontier"] = await self._perform_frontier(
                zm, hrm_matrix, tiny_matrix, run_id
            )
        except Exception as e:
            self.logger.log("FrontierError", {"run_id": run_id, "error": str(e)})
            results["frontier"] = {"error": str(e)}

        # --- Inter-model Δ analysis ---
        try:
            results["delta_analysis"] = await self._perform_delta_analysis(
                zm, hrm_matrix, tiny_matrix, hrm_names, tiny_names, run_id
            )
        except Exception as e:
            self.logger.log("DeltaAnalysisError", {"run_id": run_id, "error": str(e)})
            results["delta_analysis"] = {"error": str(e)}

        # --- Intensity report ---
        try:
            results["intensity"] = await self._generate_intensity_report(
                zm, hrm_matrix, tiny_matrix, hrm_names, tiny_names, run_id
            )
        except Exception as e:
            self.logger.log("IntensityError", {"run_id": run_id, "error": str(e)})
            results["intensity"] = {"error": str(e)}

        # --- PHOS-guarded analysis ---
        try:
            results["phos"] = await self._perform_phos_analysis(run_id)
        except Exception as e:
            self.logger.log("PHOSAnalysisError", {"run_id": run_id, "error": str(e)})
            results["phos"] = {"error": str(e)}

        # --- SCM visuals (radar, delta bars, histograms, scatter) ---
        try:
            # prefer already-in-memory matrices from scoring_results
            hrm_scm = scoring_results.get("hrm_scm_matrix")
            tiny_scm = scoring_results.get("tiny_scm_matrix")
            scm_names = scoring_results.get("scm_names", [])
            if hrm_scm is None or tiny_scm is None or not scm_names:
                # fallback: read aligned versions from storage if present
                storage = self.container.get("gap_storage")
                aligned = storage.base_dir / run_id / "aligned"
                hrm_scm = np.load(aligned / "hrm_scm_matrix.npy")
                tiny_scm = np.load(aligned / "tiny_scm_matrix.npy")
                import json
                with open(aligned / "hrm_scm_metric_names.json", "r", encoding="utf-8") as f:
                    scm_names = json.load(f)

            scm_dir = self.config.base_dir / run_id / "visuals" / "scm"
            img_paths = render_scm_images(hrm_scm, tiny_scm, scm_names, scm_dir)
            results["scm_visuals"] = img_paths
        except Exception as e:
            self.logger.log("SCMVisualsError", {"run_id": run_id, "error": str(e)})
            results["scm_visuals"] = {"error": str(e)}



        # --- Topology (UMAP/DBSCAN + homology) ---
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
        except Exception as e:
            self.logger.log("TopologyError", {"run_id": run_id, "error": str(e)})
            results["topology"] = {"error": str(e)}

        return results

    # -------------------------------------------------------------------------
    # Resolution of matrices/names (native first, else SCM)
    # -------------------------------------------------------------------------
    def _resolve_mats_and_names(
        self, scoring_results: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """
        Prefer native alignment if present and shapes match; else fall back to SCM.
        Returns: (hrm_matrix, tiny_matrix, hrm_names, tiny_names)
        """
        H = scoring_results.get("hrm_vectors")
        T = scoring_results.get("tiny_vectors")

        if isinstance(H, np.ndarray) and isinstance(T, np.ndarray):
            if H.shape == T.shape and H.shape[1] > 0:
                hrm_names = list(scoring_results.get("hrm_names", []))
                tiny_names = list(scoring_results.get("tiny_names", []))
                return H, T, hrm_names, tiny_names
            else:
                use_scm = True
        else:
            use_scm = True

        if use_scm:
            Hs = scoring_results.get("hrm_scm_matrix")
            Ts = scoring_results.get("tiny_scm_matrix")
            if not (isinstance(Hs, np.ndarray) and isinstance(Ts, np.ndarray) and Hs.shape == Ts.shape):
                raise ValueError("SCM fallback unavailable or shape-mismatched.")
            names = list(scoring_results.get("scm_names", []))
            return Hs, Ts, names, names

        # Defensive (should not hit)
        raise ValueError("Unable to resolve matrices/names for analysis.")

    # -------------------------------------------------------------------------
    # Individual steps
    # -------------------------------------------------------------------------
    async def _perform_frontier(
        self,
        zm,
        hrm_matrix: np.ndarray,
        tiny_matrix: np.ndarray,
        run_id: str,
    ) -> Dict[str, Any]:
        """Render the frontier (aligned difference field). Saves into visuals/."""
        if hrm_matrix.shape != tiny_matrix.shape:
            raise ValueError(f"Frontier: shape mismatch hrm={hrm_matrix.shape} tiny={tiny_matrix.shape}")

        out_dir = str(self.config.base_dir / run_id / "visuals")
        meta = zm.render_frontier_map(
            hrm_matrix,
            tiny_matrix,
            out_dir=out_dir,
            pos_label="HRM",
            neg_label="Tiny",
            k_latent=20,
        )
        return meta

    async def _perform_delta_analysis(
        self,
        zm,
        hrm_matrix: np.ndarray,
        tiny_matrix: np.ndarray,
        hrm_names: list[str],
        tiny_names: list[str],
        run_id: str,
    ) -> Dict[str, Any]:
        """Perform inter-model delta analysis (numbers for blog)."""
        if hrm_matrix.shape != tiny_matrix.shape:
            raise ValueError(f"Delta: shape mismatch hrm={hrm_matrix.shape} tiny={tiny_matrix.shape}")

        out_dir = str(self.config.base_dir / run_id / "metrics")
        delta_meta = zm.render_intermodel_delta(
            hrm_matrix,
            tiny_matrix,
            names_A=hrm_names,
            names_B=tiny_names,
            output_dir=out_dir,
            pos_label="HRM",
            neg_label="Tiny",
        )

        # Optional quick |Δ| image for eyeballing
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
            delta_meta["delta_abs_heat"] = str(png)
        except Exception:
            pass

        return delta_meta

    async def _generate_intensity_report(
        self,
        zm,
        hrm_matrix: np.ndarray,
        tiny_matrix: np.ndarray,
        hrm_names: list[str],
        tiny_names: list[str],
        run_id: str,
    ) -> Dict[str, Any]:
        """Generate intensity report into metrics/."""
        out_dir = str(self.config.base_dir / run_id / "metrics")
        intensity = zm.build_intensity_report(
            hrm_matrix=hrm_matrix,
            tiny_matrix=tiny_matrix,
            hrm_metric_names=hrm_names,
            tiny_metric_names=tiny_names,
            out_dir=out_dir,
            top_k=25,
        )
        return intensity

    async def _perform_phos_analysis(self, run_id: str) -> Dict[str, Any]:
        """Perform PHOS-guarded analysis → visuals/vpm_* and metrics JSON."""
        from stephanie.zeromodel.vpm_phos import build_hrm_vs_tiny_guarded

        df_proj = self._prepare_phos_data(run_id)
        if df_proj.empty:
            return {"status": "no_rows_for_df"}

        missing_dims = [
            d for d in self.config.dimensions
            if f"hrm.{d}" not in df_proj.columns or f"tiny.{d}" not in df_proj.columns
        ]
        if missing_dims:
            self.logger.log("PHOSMissingDims", {"run_id": run_id, "missing_dims": missing_dims})

        out_prefix = str(self.config.base_dir / run_id / "visuals" / "vpm")
        phos_res = build_hrm_vs_tiny_guarded(
            df_proj,
            dimensions=self.config.dimensions,
            out_prefix=out_prefix,
            tl_fracs=(0.25, 0.16, 0.36, 0.09),
            delta=0.02,
            interleave=self.config.interleave,
            weights=None,
        )
        return phos_res

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _prepare_phos_data(self, run_id: str):
        """
        Load rows_for_df.{parquet,csv} written by ScoringProcessor and keep
        canonical hrm./tiny. columns (plus node_id).
        """
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

        keep = ["node_id"] + [
            c for c in df.columns
            if isinstance(c, str) and (c.startswith("hrm.") or c.startswith("tiny."))
        ]
        return df[keep]
