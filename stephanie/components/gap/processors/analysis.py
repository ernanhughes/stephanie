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
    """
    Orchestrates analytical computations for GAP system.
    
    Handles the complete analytical pipeline including frontier mapping, delta analysis,
    intensity reporting, PHOS-guarded analysis, and topological feature detection.
    Supports both native vector alignment and SCM matrix fallback strategies.
    """

    def __init__(self, config: GapConfig, container, logger):
        """Initialize analysis processor with configuration and dependencies"""
        self.config = config
        self.container = container
        self.logger = logger
        self.topology_processor: TopologyProcessor | None = None
        _logger.debug(f"AnalysisProcessor initialized with config: {config}")

    # --- Keep orchestrator API happy ---
    async def execute_analysis(
        self,
        scoring_results: Dict[str, Any],
        run_id: str,
        manifest: Any | None = None,
    ) -> Dict[str, Any]:
        """API-compatible wrapper for perform_analysis"""
        return await self.perform_analysis(scoring_results, run_id, manifest)

    async def perform_analysis(
        self,
        scoring_results: Dict[str, Any],
        run_id: str,
        manifest: Any | None = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analytical pipeline on aligned HRM/Tiny matrices.
        
        Analysis flow:
        1. Frontier mapping for difference field visualization
        2. Delta analysis for inter-model performance gaps
        3. Intensity reporting for metric variance analysis
        4. PHOS-guarded statistical analysis
        5. SCM visualization generation
        6. Topological analysis with UMAP and persistent homology
        
        Prefers native alignment when shapes match; falls back to SCM matrices otherwise.
        """
        _logger.debug(f"Starting analysis pipeline for run {run_id}")
        _logger.debug(f"Scoring results keys: {list(scoring_results.keys())}")
        
        zm = self.container.get("zeromodel")

        # Setup output directories
        out_metrics_dir = self.config.base_dir / run_id / "metrics"
        out_visuals_dir = self.config.base_dir / run_id / "visuals"
        out_metrics_dir.mkdir(parents=True, exist_ok=True)
        out_visuals_dir.mkdir(parents=True, exist_ok=True)
        _logger.debug(f"Created output directories: {out_metrics_dir}, {out_visuals_dir}")

        # Resolve matrices/names once for consistency across analyses
        hrm_matrix, tiny_matrix, hrm_names, tiny_names = self._resolve_mats_and_names(scoring_results)
        _logger.debug(f"Resolved matrices - HRM: {hrm_matrix.shape}, Tiny: {tiny_matrix.shape}")
        _logger.debug(f"HRM names: {len(hrm_names)}, Tiny names: {len(tiny_names)}")

        results: Dict[str, Any] = {}

        # --- Frontier Analysis: Visualize difference field ---
        _logger.debug("Starting frontier analysis...")
        try:
            results["frontier"] = await self._perform_frontier(
                zm, hrm_matrix, tiny_matrix, run_id, 
                scoring_results.get("hrm_label", "HRM"), 
                scoring_results.get("tiny_label", "Tiny")
            )
            _logger.debug("Frontier analysis completed successfully")
        except Exception as e:
            self.logger.log("FrontierError", {"run_id": run_id, "error": str(e)})
            _logger.error(f"Frontier analysis failed: {e}", exc_info=True)
            results["frontier"] = {"error": str(e)}

        # --- Inter-model Delta Analysis: Quantify performance gaps ---
        _logger.debug("Starting delta analysis...")
        try:
            results["delta_analysis"] = await self._perform_delta_analysis(
                zm, hrm_matrix, tiny_matrix, hrm_names, tiny_names, run_id
            )
            _logger.debug("Delta analysis completed successfully")
        except Exception as e:
            self.logger.log("DeltaAnalysisError", {"run_id": run_id, "error": str(e)})
            _logger.error(f"Delta analysis failed: {e}", exc_info=True)
            results["delta_analysis"] = {"error": str(e)}

        # --- Intensity Reporting: Identify high-variance metrics ---
        _logger.debug("Starting intensity report generation...")
        try:
            results["intensity"] = await self._generate_intensity_report(
                zm, hrm_matrix, tiny_matrix, hrm_names, tiny_names, run_id
            )
            _logger.debug("Intensity report generated successfully")
        except Exception as e:
            self.logger.log("IntensityError", {"run_id": run_id, "error": str(e)})
            _logger.error(f"Intensity report generation failed: {e}", exc_info=True)
            results["intensity"] = {"error": str(e)}

        # --- PHOS Analysis: Guarded statistical analysis ---
        _logger.debug("Starting PHOS analysis...")
        try:
            results["phos"] = await self._perform_phos_analysis(run_id)
            _logger.debug("PHOS analysis completed successfully")
        except Exception as e:
            self.logger.log("PHOSAnalysisError", {"run_id": run_id, "error": str(e)})
            _logger.error(f"PHOS analysis failed: {e}", exc_info=True)
            results["phos"] = {"error": str(e)}

        # --- SCM Visualization: Generate comprehensive visualizations ---
        _logger.debug("Starting SCM visualization generation...")
        try:
            # Prefer already-in-memory matrices from scoring_results for efficiency
            hrm_scm = scoring_results.get("hrm_scm_matrix")
            tiny_scm = scoring_results.get("tiny_scm_matrix")
            scm_names = scoring_results.get("scm_names", [])
            
            # Fallback to stored matrices if memory versions unavailable
            if hrm_scm is None or tiny_scm is None or not scm_names:
                _logger.debug("Memory SCM matrices not found, loading from storage...")
                storage = self.container.get("gap_storage")
                aligned = storage.base_dir / run_id / "aligned"
                hrm_scm = np.load(aligned / "hrm_scm_matrix.npy")
                tiny_scm = np.load(aligned / "tiny_scm_matrix.npy")
                import json
                with open(aligned / "hrm_scm_metric_names.json", "r", encoding="utf-8") as f:
                    scm_names = json.load(f)
                _logger.debug(f"Loaded SCM matrices from storage: {hrm_scm.shape}, {tiny_scm.shape}")

            scm_dir = self.config.base_dir / run_id / "visuals" / "scm"
            pos_label = scoring_results.get("hrm_label", "HRM")
            neg_label = scoring_results.get("tiny_label", "Tiny")
            img_paths = render_scm_images(hrm_scm, tiny_scm, scm_names, scm_dir, pos_label, neg_label)  
            results["scm_visuals"] = img_paths
            _logger.debug(f"Generated {len(img_paths)} SCM visualizations")
        except Exception as e:
            self.logger.log("SCMVisualsError", {"run_id": run_id, "error": str(e)})
            _logger.error(f"SCM visualization generation failed: {e}", exc_info=True)
            results["scm_visuals"] = {"error": str(e)}

        # --- Topology Analysis: UMAP + Persistent Homology ---
        _logger.debug("Starting topology analysis...")
        try:
            self.topology_processor = TopologyProcessor(
                TopologyConfig(
                    use_weighted=True,
                    weights={
                        "reasoning.score01": 1.3,  # Emphasize reasoning dimension
                        "knowledge.score01": 1.1,  # Slightly emphasize knowledge
                        "clarity.score01": 1.0,    # Baseline weight for clarity
                        "faithfulness.score01": 1.2, # Emphasize faithfulness
                        "coverage.score01": 1.0,   # Baseline weight for coverage
                    },
                    umap_n_neighbors=15,    # Balance local/global structure
                    umap_min_dist=0.2,      # Allow some clustering
                    dbscan_eps=0.3,         # Cluster proximity threshold
                    dbscan_min_samples=5,   # Minimum cluster size
                    max_betti_dim=1,        # Analyze loops (1D homology)
                ),
                container=self.container,
                logger=self.logger,
            )
            topo_out = await self.topology_processor.run(run_id, base_dir=self.config.base_dir)
            results["topology"] = topo_out
            _logger.debug("Topology analysis completed successfully")
        except Exception as e:
            self.logger.log("TopologyError", {"run_id": run_id, "error": str(e)})
            _logger.error(f"Topology analysis failed: {e}", exc_info=True)
            results["topology"] = {"error": str(e)}

        _logger.info(f"Analysis pipeline completed for run {run_id} with {len(results)} components")
        return results

    # -------------------------------------------------------------------------
    # Matrix Resolution: Native vs SCM Fallback
    # -------------------------------------------------------------------------

    def _resolve_mats_and_names(
        self, scoring_results: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """
        Resolve analysis matrices with preference for native alignment.
        
        Strategy:
        1. Prefer native vectors if shapes match and dimensions are valid
        2. Fall back to SCM matrices if native unavailable or mismatched
        3. Ensure consistent naming across both strategies
        
        Returns:
            Tuple of (hrm_matrix, tiny_matrix, hrm_names, tiny_names)
            
        Raises:
            ValueError: When no suitable matrices can be resolved
        """
        _logger.debug("Resolving matrices for analysis...")
        
        # Try native vector alignment first
        H = scoring_results.get("hrm_vectors")
        T = scoring_results.get("tiny_vectors")

        if isinstance(H, np.ndarray) and isinstance(T, np.ndarray):
            _logger.debug(f"Found native vectors - HRM: {H.shape}, Tiny: {T.shape}")
            if H.shape == T.shape and H.shape[1] > 0:
                hrm_names = list(scoring_results.get("hrm_names", []))
                tiny_names = list(scoring_results.get("tiny_names", []))
                _logger.debug("Using native vector alignment (shapes match)")
                return H, T, hrm_names, tiny_names
            else:
                _logger.warning(f"Native vector shape mismatch: HRM {H.shape} != Tiny {T.shape}")
                use_scm = True
        else:
            _logger.debug("Native vectors not available, falling back to SCM")
            use_scm = True

        # Fall back to SCM matrices
        if use_scm:
            Hs = scoring_results.get("hrm_scm_matrix")
            Ts = scoring_results.get("tiny_scm_matrix")
            if not (isinstance(Hs, np.ndarray) and isinstance(Ts, np.ndarray) and Hs.shape == Ts.shape):
                raise ValueError("SCM fallback unavailable or shape-mismatched.")
            names = list(scoring_results.get("scm_names", []))
            _logger.debug(f"Using SCM matrices: {Hs.shape} with {len(names)} metrics")
            return Hs, Ts, names, names

        # Should not reach here with proper error handling above
        raise ValueError("Unable to resolve matrices/names for analysis.")

    # -------------------------------------------------------------------------
    # Individual Analytical Components
    # -------------------------------------------------------------------------

    async def _perform_frontier(
        self,
        zm,
        pos_matrix: np.ndarray,
        neg_matrix: np.ndarray,
        pos_label: str,
        neg_label: str,
        run_id: str,
    ) -> Dict[str, Any]:
        """
        Render frontier map visualizing the difference field between HRM and Tiny.
        
        Creates a 2D projection showing the performance frontier where models diverge.
        Uses latent space analysis to identify regions of maximum differentiation.
        """
        _logger.debug(f"Rendering frontier map for matrices: {pos_matrix.shape}")
        
        if pos_matrix.shape != neg_matrix.shape:
            raise ValueError(f"Frontier: shape mismatch hrm={pos_matrix.shape} tiny={neg_matrix.shape}")

        out_dir = str(self.config.base_dir / run_id / "visuals")
        meta = zm.render_frontier_map(
            pos_matrix,
            neg_matrix,
            out_dir=out_dir,
            pos_label=pos_label,
            neg_label=neg_label,
            k_latent=20,  # Number of latent dimensions for projection
        )
        _logger.debug(f"Frontier map generated with metadata: {list(meta.keys())}")
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
        """
        Perform inter-model delta analysis for performance gap quantification.
        
        Computes:
        - Absolute differences |HRM - Tiny|
        - Statistical significance of gaps
        - Top diverging metrics
        - Visual heatmap of differences
        """
        _logger.debug("Starting inter-model delta analysis...")
        
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
        _logger.debug(f"Delta analysis completed with {len(delta_meta)} metrics")

        # Generate quick visual heatmap for rapid inspection
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
            _logger.debug(f"Delta heatmap saved to: {png}")
        except Exception as e:
            _logger.warning(f"Could not generate delta heatmap: {e}")

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
            top_k=25,  # Number of top metrics to highlight
        )
        _logger.debug(f"Intensity report generated with {len(intensity)} entries")
        return intensity

    async def _perform_phos_analysis(self, run_id: str) -> Dict[str, Any]:
        """
        Perform PHOS (Probabilistic Hierarchical Outcome Scoring) guarded analysis.
        
        PHOS provides:
        - Statistical guarding against false discoveries
        - Threshold learning for significance determination
        - Multi-dimensional performance profiling
        - Confidence-weighted conclusions
        """
        _logger.debug("Starting PHOS-guarded analysis...")
        from stephanie.zeromodel.vpm_phos import build_hrm_vs_tiny_guarded

        df_proj = self._prepare_phos_data(run_id)
        if df_proj.empty:
            _logger.warning(f"No PHOS data available for run {run_id}")
            return {"status": "no_rows_for_df"}

        # Validate dimension availability
        missing_dims = [
            d for d in self.config.dimensions
            if f"hrm.{d}" not in df_proj.columns or f"tiny.{d}" not in df_proj.columns
        ]
        if missing_dims:
            self.logger.log("PHOSMissingDims", {"run_id": run_id, "missing_dims": missing_dims})
            _logger.warning(f"Missing dimensions for PHOS analysis: {missing_dims}")

        out_prefix = str(self.config.base_dir / run_id / "visuals" / "vpm")
        phos_res = build_hrm_vs_tiny_guarded(
            df_proj,
            dimensions=self.config.dimensions,
            out_prefix=out_prefix,
            tl_fracs=(0.25, 0.16, 0.36, 0.09),  # Threshold learning fractions
            delta=0.02,                          # Minimum detectable effect size
            interleave=self.config.interleave,   # Interleaving strategy
            weights=None,                        # Optional dimension weighting
        )
        _logger.debug(f"PHOS analysis completed with {len(phos_res)} results")
        return phos_res

    # -------------------------------------------------------------------------
    # Data Preparation Helpers
    # -------------------------------------------------------------------------

    def _prepare_phos_data(self, run_id: str):
        """
        Load and prepare data frame for PHOS analysis.
        
        Loads the canonical scoring data with HRM and Tiny columns,
        preserving only the essential columns for analysis.
        
        Returns:
            DataFrame with node_id, hrm.*, and tiny.* columns
        """
        _logger.debug(f"Preparing PHOS data for run {run_id}")
        import pandas as pd

        storage = self.container.get("gap_storage")
        raw_dir = storage.base_dir / run_id / "raw"

        # Try parquet first for performance, fall back to CSV
        pq = raw_dir / "rows_for_df.parquet"
        csv = raw_dir / "rows_for_df.csv"

        if pq.exists():
            df = pd.read_parquet(pq)
            _logger.debug(f"Loaded PHOS data from parquet: {df.shape}")
        elif csv.exists():
            df = pd.read_csv(csv)
            _logger.debug(f"Loaded PHOS data from CSV: {df.shape}")
        else:
            self.logger.log("PHOSRowsMissing", {"run_id": run_id, "raw_dir": str(raw_dir)})
            _logger.error(f"No PHOS data files found in {raw_dir}")
            return pd.DataFrame()

        # Filter to essential columns: node_id and model-specific scores
        keep = ["node_id"] + [
            c for c in df.columns
            if isinstance(c, str) and (c.startswith("hrm.") or c.startswith("tiny."))
        ]
        filtered_df = df[keep]
        _logger.debug(f"Filtered PHOS data to {len(keep)} columns")
        
        return filtered_df