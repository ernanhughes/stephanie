# stephanie/components/gap/orchestrator.py
from __future__ import annotations

import logging
from typing import Any, Dict, Callable

from stephanie.components.gap.models import GapConfig
from stephanie.components.gap.io.storage import GapStorageService
from stephanie.components.gap.io.manifest import ManifestManager
from stephanie.components.gap.io.data_retriever import DataRetriever, RetrieverConfig
from stephanie.components.gap.processors.scoring import ScoringProcessor
from stephanie.components.gap.processors.analysis import AnalysisProcessor
from stephanie.components.gap.processors.calibration import CalibrationProcessor
from stephanie.components.gap.processors.report import ReportBuilder
from stephanie.components.gap.services.scm_term_head import SCMTermHeadService

# NEW: significance stage
from stephanie.components.gap.processors.significance import (
    SignificanceProcessor,
    SignificanceConfig,
)

_logger = logging.getLogger(__name__)


class GapAnalysisOrchestrator:
    def __init__(self, config: GapConfig, container, logger, memory=None):
        self.config = config
        self.container = container
        self.logger = logger
        self.memory = memory

        # ---- Storage as a proper Service ----
        try:
            self.container.register(
                name="gap_storage",
                factory=lambda: GapStorageService(),
                dependencies=[],
                init_args={"base_dir": str(self.config.base_dir), "logger": self.logger},
            )
        except ValueError:
            pass  # already registered

        # Optional: shared SCM term head (used in scoring if enabled)
        if config.enable_scm_head:
            try:
                container.register(
                    name="scm_term_head",
                    factory=lambda: SCMTermHeadService(),
                    dependencies=[],
                    init_args={"config": config.scm, "logger": logger},
                )
            except ValueError:
                pass

        # Ensure it's initialized and available
        self.storage = self.container.get("gap_storage")
        self.manifest_manager = ManifestManager(self.storage)

        # ---- Processors ----
        self.scoring_processor = ScoringProcessor(self.config, container, logger)
        self.analysis_processor = AnalysisProcessor(self.config, container, logger)
        self.calibration_processor = CalibrationProcessor(self.config, container, logger)

        # NEW: stats/significance lives in its own processor
        self.significance_processor = SignificanceProcessor(
            SignificanceConfig(
                n_nulls=getattr(self.config, "n_nulls", 100),
                n_bootstrap=getattr(self.config, "n_bootstrap", 50),
                random_seed=getattr(self.config, "random_seed", 42),
                max_betti_dim=1,
            ),
            logger=self.logger,
        )

        # ---- Data retriever (memory-backed by default) ----
        safe_limit = self.config.per_dim_cap if self.config.per_dim_cap is not None else 10**9
        self.retriever = DataRetriever(
            container,
            logger,
            retriever_cfg=RetrieverConfig(
                source="memory",
                limit=safe_limit,   # always an int
            ),
        )

    # ---- progress helpers ---------------------------------------------------
    def _mk_progress_cb(self, stage: str, every: int = 10):
        """
        Returns a callback that accepts EITHER:
        - (done:int, total:int, extras:dict|None)
        - (substage:str, done:int, total:int, extras:dict|None)
        and logs at most ~every% (plus the final tick).
        """
        state = {"last_bucket": -1}

        def _cb(*args):
            # Parse args
            if not args:
                return

            substage = None
            extras = None

            if isinstance(args[0], str):
                # ("substage", done, total, [extras])
                substage = args[0]
                done  = int(args[1]) if len(args) > 1 and args[1] is not None else 0
                total = int(args[2]) if len(args) > 2 and args[2] is not None else 1
                extras = args[3] if len(args) > 3 else None
            else:
                # (done, total, [extras])
                done  = int(args[0]) if len(args) > 0 and args[0] is not None else 0
                total = int(args[1]) if len(args) > 1 and args[1] is not None else 1
                extras = args[2] if len(args) > 2 else None

            total = max(1, total)  # guard
            pct = int((done / total) * 100)
            bucket = pct // max(1, every)

            if bucket != state["last_bucket"] or done == total:
                payload = {
                    "stage": stage,
                    "done": done,
                    "total": total,
                    "percent": pct,
                }
                if substage:
                    payload["substage"] = substage
                if extras:
                    payload.update(extras)

                self.logger.log("GapProgress", payload)
                state["last_bucket"] = bucket

        return _cb

    # ---- the end-to-end run -------------------------------------------------
    async def execute_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = context.get("pipeline_run_id") or context.get("run_id") or "gap_run"
        dataset_name = context.get("dataset", "unknown")

        # Manifest
        m = self.manifest_manager.start_run(
            run_id=run_id,
            dataset=dataset_name,
            models={"hrm": self.config.hrm_scorers[0], "tiny": self.config.tiny_scorers[0]},
        )
        self.manifest_manager.attach_dimensions(run_id, self.config.dimensions)

        # 1) Data
        triples_by_dim = await self.retriever.get_triples_by_dimension(
            self.config.dimensions,
            memory=self.memory,
            limit=self.retriever.cfg.limit,
        )

        # 2) Scoring (+timeline) with progress callback
        score_out = await self.scoring_processor.execute_scoring(
            triples_by_dim,
            run_id,
            manifest=m,
            progress_cb=self._mk_progress_cb(stage="scoring", every=10),
        )

        # 3) Analysis (frontier, delta, intensity, PHOS, topology incl. betti.json)
        analysis_out = await self.analysis_processor.execute_analysis(score_out, run_id)

        # 4) Significance (p-values, CIs, nulls, sensitivity, assumptions)
        #    Note: SignificanceProcessor expects the Δ used by topology saved at:
        #    aligned/delta_core5.npy and reads metrics/betti.json.
        try:
            significance_out = await self.significance_processor.run(run_id, base_dir=self.config.base_dir)
        except Exception as e:
            self.logger.log("SignificanceStageError", {"run_id": run_id, "error": str(e)})
            significance_out = {"status": "error", "error": str(e)}

        # 5) Calibration (uses analysis outputs; independent of significance)
        calib_out = await self.calibration_processor.execute_calibration(analysis_out, run_id)

        # 6) Report (Markdown quicklook) — pass significance so it can be shown
        reporter = ReportBuilder(self.config, self.container, self.logger)
        analysis_out = {**analysis_out, "significance": significance_out}
        report_out = await reporter.build(run_id, analysis_out, score_out)

        result = {
            "run_id": run_id,
            "score": score_out,
            "analysis": analysis_out,
            "significance": significance_out,
            "calibration": calib_out,
            "report": report_out,
        }
        self.manifest_manager.finish_run(run_id, result)
        return result
