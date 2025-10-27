# stephanie/components/gap/orchestrator.py
"""
GAP Analysis Orchestrator - Core pipeline coordinator for model comparison.

This module implements the main orchestration logic that coordinates the entire
GAP analysis pipeline from data retrieval through scoring, analysis, significance
testing, calibration, and reporting.

The orchestrator manages the workflow:
1. Data Preparation → 2. Model Scoring → 3. Analysis → 4. Significance Testing → 
5. Calibration → 6. Reporting

It ensures proper dependency injection, error handling, and progress tracking
across all pipeline stages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.components.gap.io.data_retriever import (DataRetriever,
                                                        RetrieverConfig)
from stephanie.components.gap.io.manifest import ManifestManager
from stephanie.components.gap.io.storage import GapStorageService
from stephanie.components.gap.models import GapConfig
from stephanie.components.gap.processors.analysis import AnalysisProcessor
from stephanie.components.gap.processors.calibration import \
    CalibrationProcessor
from stephanie.components.gap.processors.report import ReportBuilder
from stephanie.components.gap.processors.scoring import ScoringProcessor
from stephanie.components.gap.processors.significance import (
    SignificanceConfig, SignificanceProcessor)
from stephanie.components.gap.services.epistemic_guard_service import (
    EGVisualService, EpistemicGuardService)
from stephanie.components.gap.services.risk_predictor_service import \
    RiskPredictorService
from stephanie.components.gap.services.scm_service import SCMService
from stephanie.utils.progress_mixin import ProgressMixin

_logger = logging.getLogger(__name__)


class GapAnalysisOrchestrator(ProgressMixin):
    """
    Main orchestrator for GAP analysis pipeline execution.
    
    Coordinates the entire workflow from data preparation through final reporting,
    managing dependencies between processors and ensuring consistent error handling
    and progress tracking across all stages.
    
    Responsibilities:
    - Dependency injection and service registration
    - Pipeline stage sequencing and coordination
    - Progress tracking and manifest management
    - Error handling and recovery
    - Result aggregation and final reporting
    """
    
    def __init__(self, cfg: GapConfig, container, logger, memory=None):
        """
        Initialize the GAP analysis orchestrator with all required components.
        
        Args:
            config: GAP configuration parameters
            container: Dependency injection container for service management
            logger: Structured logging interface
            memory: Optional memory interface for data access (uses container if None)
            
        Initializes:
            - Storage service for artifact persistence
            - SCM service for Shared Core Metrics
            - All analysis processors (scoring, analysis, calibration, significance)
            - Data retriever for sample collection
            - Progress tracking system
        """
        self.cfg = cfg
        self.container = container
        self.logger = logger
        self.memory = memory

        # ---- Storage as a proper Service ----
        # Register and initialize persistent storage for analysis artifacts
        try:
            self.container.register(
                name="gap_storage",
                factory=lambda: GapStorageService(),
                dependencies=[],
                init_args={
                    "base_dir": str(self.cfg.base_dir),
                    "logger": self.logger,
                },
            )
        except ValueError:
            pass  # Already registered - safe to ignore

        # Optional: shared SCM term head (used in scoring if enabled)
        # Enables Shared Core Metrics projection for cross-model alignment
        if cfg.enable_scm_head:
            try:
                container.register(
                    name="scm_service",
                    factory=lambda: SCMService(),
                    dependencies=[],
                    init_args={"config": cfg.scm, "logger": logger},
                )
            except ValueError:
                pass  # Service already registered


            try:
                container.register(
                    name="ep_guard",
                    factory=lambda: EpistemicGuardService(),
                    dependencies=[],
                    init_args={"config": {"out_dir": str(cfg.base_dir / "eg"), "thresholds": (0.2, 0.6)}, "logger": logger},
                )
                container.register(
                    name="eg_visual",
                    factory=lambda: EGVisualService(),
                    dependencies=[],
                    init_args={"config": {"out_dir": str(cfg.base_dir / "eg" / "img")}, "logger": logger},
                )
            except ValueError:
                pass


            # ---- Risk Predictor (single source of truth for risk & thresholds) ----
            try:
                self.container.register(
                    name="risk_predictor",
                    factory=lambda: RiskPredictorService(cfg=cfg, memory=memory, logger=logger),
                    dependencies=["?memcube"],  # optional
                    init_args={
                        "config": {
                            "bundle_path": "./models/risk/bundle.joblib",
                            "default_domains": ("science", "history", "geography", "tech", "general"),
                            "calib_ttl_s": 3600,
                            "fallback_low": 0.20,
                            "fallback_high": 0.60,
                            # optionally inject a memcube client explicitly:
                            # "memcube": container.get("memcube"),
                        }
                    },
                )
            except ValueError:
                # already registered elsewhere (e.g., global bootstrap) — safe
                pass


        # Ensure storage is initialized and available to all components
        self.storage = self.container.get("gap_storage")



        self.manifest_manager = ManifestManager(self.storage)

        # ---- Processor Initialization ----
        # Each processor handles a specific stage of the analysis pipeline
        
        # Scoring: Runs HRM and Tiny models on all samples, produces timelines
        self.scoring_processor = ScoringProcessor(
            self.cfg, container, logger
        )
        
        # Analysis: Computes delta fields, topology, frontier maps, PHOS packs
        self.analysis_processor = AnalysisProcessor(
            self.cfg, container, logger
        )
        
        # Calibration: Determines routing thresholds and model escalation policies
        self.calibration_processor = CalibrationProcessor(
            self.cfg, container, logger
        )

        # Significance: Statistical validation of topological findings
        # Handles p-values, confidence intervals, null hypothesis testing
        self.significance_processor = SignificanceProcessor(
            SignificanceConfig(
                n_nulls=getattr(self.cfg, "n_nulls", 100),
                n_bootstrap=getattr(self.cfg, "n_bootstrap", 50),
                random_seed=getattr(self.cfg, "random_seed", 42),
                max_betti_dim=1,  # Focus on H1 loops (1-dimensional holes)
            ),
            logger=self.logger,
        )

        # ---- Data retriever (memory-backed by default) ----
        # Handles sample collection with safety limits to prevent memory issues
        safe_limit = self.cfg.per_dim_cap if self.cfg.per_dim_cap is not None else 10**9
        self.retriever = DataRetriever(
            container,
            logger,
            retriever_cfg=RetrieverConfig(
                source="memory",  # Options: "memory", "database", "file"
                limit=safe_limit,   # Safety cap per dimension
            ),
        )
        
        # Initialize progress tracking system
        self._init_progress(container, logger)

    # ---- the end-to-end run -------------------------------------------------
    async def execute_analysis(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the complete GAP analysis pipeline.
        
        Pipeline Stages:
        1. Data Preparation: Retrieve and validate samples by dimension
        2. Model Scoring: Run HRM and Tiny models, generate timelines  
        3. Analysis: Compute delta fields, topology, frontier maps
        4. Significance: Statistical validation of topological findings
        5. Calibration: Determine routing thresholds and escalation policies
        6. Reporting: Generate comprehensive analysis report
        
        Args:
            context: Pipeline execution context containing:
                - pipeline_run_id: Unique identifier for this run
                - dataset: Dataset name or identifier
                - Additional pipeline-specific parameters
                
        Returns:
            Dictionary containing complete analysis results:
            {
                "run_id": str,                    # Analysis run identifier
                "score": Dict,                    # Scoring stage results
                "analysis": Dict,                 # Analysis stage results  
                "significance": Dict,             # Statistical significance
                "calibration": Dict,              # Routing thresholds
                "report": Dict,                   # Generated report
                "manifest": Dict                  # Complete run manifest
            }
            
        Raises:
            AnalysisError: If any pipeline stage fails critically
            ConfigurationError: If required context is missing
            
        Logs:
            - Progress events for each pipeline stage
            - Error events with detailed context for debugging
            - Completion metrics for monitoring and validation
        """
        # Extract run identifiers from context
        run_id = (
            context.get("pipeline_run_id")
            or context.get("run_id")
            or "gap_run"
        )
        dataset_name = context.get("dataset", "unknown")

        # Initialize manifest for run tracking and artifact management
        m = self.manifest_manager.start_run(
            run_id=run_id,
            dataset=dataset_name,
            models={
                "hrm": self.cfg.hrm_scorers[0],
                "tiny": self.cfg.tiny_scorers[0],
            },
        )
        self.manifest_manager.attach_dimensions(run_id, self.cfg.dimensions)

        # Stage 1: Data Preparation
        # Retrieve conversation turns organized by reasoning dimension
        self.pstart(task=f"data:{run_id}", total=1, meta={"dataset": dataset_name})
        triples_by_dim = await self.retriever.get_triples_by_dimension(
            self.cfg.dimensions,
            memory=self.memory,
            limit=self.retriever.cfg.limit,
        )
        self.pdone(task=f"data:{run_id}", extra={"dims": len(triples_by_dim)})

        # Stage 2: Model Scoring 
        # Run HRM and Tiny models on all samples, generate VPM timelines
        score_out = await self.scoring_processor.execute_scoring(
            triples_by_dim,
            run_id,
            manifest=m,
        )

        # Stage 3: Analysis
        # Compute delta fields, persistent homology, frontier maps, PHOS packs
        analysis_out = await self.analysis_processor.execute_analysis(
            score_out,
            run_id,
            manifest=m,
        )

        # Stage 4: Significance Testing
        # Statistical validation of topological findings against null models
        try:
            significance_out = await self.significance_processor.run(
                run_id,
                base_dir=self.cfg.base_dir,
            )
        except Exception as e:
            self.logger.log(
                "SignificanceStageError", {"run_id": run_id, "error": str(e)}
            )
            significance_out = {"status": "error", "error": str(e)}

        # Stage 5: Calibration
        # Determine routing thresholds and model escalation policies
        alias_a = score_out.get("alias_a", "HRM")
        alias_b = score_out.get("alias_b", "Tiny")
        calib_out = await self.calibration_processor.execute_calibration(
            analysis_out,
            run_id,
            alias_a=alias_a,
            alias_b=alias_b,
        )

        # Stage 6: Reporting
        # Generate comprehensive Markdown report with all findings
        reporter = ReportBuilder(self.cfg, self.container, self.logger)
        # Include significance results in final analysis output
        analysis_out = {**analysis_out, "significance": significance_out}
        report_out = await reporter.build(
            run_id,
            analysis_out, 
            score_out,
        )

        # Aggregate all results and finalize run manifest
        result = {
            "run_id": run_id,
            "score": score_out,           # Scoring stage outputs
            "analysis": analysis_out,     # Analysis stage outputs  
            "significance": significance_out,  # Statistical validation
            "calibration": calib_out,     # Routing thresholds
            "report": report_out,         # Generated report
            "manifest": m.to_dict(),      # Complete run manifest
        }
        
        # Finalize manifest with complete results
        self.manifest_manager.finish_run(run_id, result)
        
        return result