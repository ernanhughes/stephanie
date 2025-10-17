# stephanie/components/gap/processors/analysis.py
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from stephanie.components.gap.models import GapConfig

logger = logging.getLogger(__name__)


class AnalysisProcessor:
    """Handles analytical computations: delta, frontier, intensity, PHOS."""

    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger

    # --- New: keep orchestrator API happy ---
    async def execute_analysis(
        self,
        scoring_results: Dict[str, Any],
        run_id: str,
        manifest: Any | None = None,
    ) -> Dict[str, Any]:
        """
        Entry point used by the orchestrator. Delegates to perform_analysis.
        """
        return await self.perform_analysis(scoring_results, run_id, manifest)

    async def perform_analysis(
        self,
        scoring_results: Dict[str, Any],
        run_id: str,
        manifest: Any | None = None,
    ) -> Dict[str, Any]:
        """
        Perform all analytical computations on aligned HRM/Tiny matrices.
        Expects scoring_results to contain:
          - 'hrm_vectors', 'tiny_vectors': np.ndarray with SAME SHAPE
          - 'hrm_names', 'tiny_names': List[str] (aligned/shared names)
        """
        zm = self.container.get("zeromodel")
        storage = self.container.get("gap_storage")

        out_metrics_dir = self.config.base_dir / run_id / "metrics"
        out_visuals_dir = self.config.base_dir / run_id / "visuals"
        out_metrics_dir.mkdir(parents=True, exist_ok=True)
        out_visuals_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Any] = {}

        # --- Frontier (aligned difference field) ---
        try:
            frontier_meta = await self._perform_frontier(zm, scoring_results, run_id)
            results["frontier"] = frontier_meta
        except Exception as e:
            self.logger.log("FrontierError", {"run_id": run_id, "error": str(e)})
            results["frontier"] = {"error": str(e)}

        # --- Inter-model Δ analysis ---
        try:
            delta_meta = await self._perform_delta_analysis(zm, scoring_results, run_id)
            results["delta_analysis"] = delta_meta
        except Exception as e:
            self.logger.log("DeltaAnalysisError", {"run_id": run_id, "error": str(e)})
            results["delta_analysis"] = {"error": str(e)}

        # --- Intensity report ---
        try:
            intensity_report = await self._generate_intensity_report(zm, scoring_results, run_id)
            results["intensity"] = intensity_report
        except Exception as e:
            self.logger.log("IntensityError", {"run_id": run_id, "error": str(e)})
            results["intensity"] = {"error": str(e)}

        # --- PHOS-guarded analysis ---
        try:
            phos_analysis = await self._perform_phos_analysis(scoring_results, run_id)
            results["phos"] = phos_analysis
        except Exception as e:
            self.logger.log("PHOSAnalysisError", {"run_id": run_id, "error": str(e)})
            results["phos"] = {"error": str(e)}

        return results

    # -------------------------------------------------------------------------
    # Individual steps
    # -------------------------------------------------------------------------
    async def _perform_frontier(self, zm, scoring_results: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """
        Render the frontier (aligned difference field). Saves into visuals/.
        """
        hrm_matrix = scoring_results["hrm_vectors"]
        tiny_matrix = scoring_results["tiny_vectors"]

        # Sanity: must be same shape already (caller aligned names)
        if hrm_matrix.shape != tiny_matrix.shape:
            raise ValueError(
                f"Frontier: shape mismatch hrm={hrm_matrix.shape} tiny={tiny_matrix.shape}"
            )

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

    async def _perform_delta_analysis(self, zm, scoring_results: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Perform inter-model delta analysis (numbers for blog)."""
        hrm_matrix = scoring_results["hrm_vectors"]
        tiny_matrix = scoring_results["tiny_vectors"]
        hrm_names = scoring_results["hrm_names"]
        tiny_names = scoring_results["tiny_names"]

        if hrm_matrix.shape != tiny_matrix.shape:
            raise ValueError(
                f"Delta: shape mismatch hrm={hrm_matrix.shape} tiny={tiny_matrix.shape}"
            )

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

    async def _generate_intensity_report(self, zm, scoring_results: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Generate intensity report into metrics/."""
        hrm_matrix = scoring_results["hrm_vectors"]
        tiny_matrix = scoring_results["tiny_vectors"]
        hrm_names = scoring_results["hrm_names"]
        tiny_names = scoring_results["tiny_names"]

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

    async def _perform_phos_analysis(self, scoring_results: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Perform PHOS-guarded analysis → visuals/vpm_* and metrics JSON."""
        from stephanie.zeromodel.vpm_phos import build_hrm_vs_tiny_guarded

        df_proj = self._prepare_phos_data(scoring_results, run_id)
        if df_proj.empty:
            return {"status": "no_rows_for_df"}

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
    def _prepare_phos_data(self, scoring_results: Dict[str, Any], run_id: str):
        """
        Load raw rows_for_df.{parquet,csv} written by ScoringProcessor and
        keep canonical hrm./tiny. columns.
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
