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
            # already registered
            pass

        # Ensure it's initialized and available
        self.storage = self.container.get("gap_storage")
        self.manifest_manager = ManifestManager(self.storage)

        # ---- Processors ----
        self.scoring_processor = ScoringProcessor(self.config, container, logger)
        self.analysis_processor = AnalysisProcessor(self.config, container, logger)
        self.calibration_processor = CalibrationProcessor(self.config, container, logger)

        # ---- Data retriever (memory-backed by default) ----
        # IMPORTANT: Never pass None for limit into memory layer
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
    def _mk_progress_cb(self, stage: str, every: int = 10) -> Callable[[int, int, Dict[str, Any]], None]:
        """
        Returns a callback like: cb(done:int, total:int, extras:dict)
        Processors can call this to emit heartbeat logs every ~`every` percent.
        """
        state = {"last_bucket": -1}
        def _cb(done: int, total: int, extras: Dict[str, Any] | None = None):
            if total <= 0:
                return
            pct = int((done / total) * 100)
            bucket = pct // max(1, every)
            if bucket != state["last_bucket"] or done == total:
                payload = {"stage": stage, "done": done, "total": total, "percent": pct}
                if extras:
                    payload.update(extras)
                self.logger.log("GapProgress", payload)
                state["last_bucket"] = bucket
        return _cb

    # ---- the end-to-end run -------------------------------------------------
    async def execute_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = context.get("pipeline_run_id") or context.get("run_id") or "gap_run"
        dataset_name = context.get("dataset", "unknown")

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
            limit=self.retriever.cfg.limit,  # guaranteed int
        )

        # 2) Scoring (+timeline) with progress callback
        score_out = await self.scoring_processor.execute_scoring(
            triples_by_dim,
            run_id,
            manifest=m,
            progress_cb=self._mk_progress_cb(stage="scoring", every=10),
        )

        # 3) Analysis (frontier, delta, intensity, PHOS)
        analysis_out = await self.analysis_processor.execute_analysis(score_out, run_id)

        # 4) Calibration (uses analysis outputs)
        calib_out = await self.calibration_processor.execute_calibration(analysis_out, run_id)

        result = {
            "run_id": run_id,
            "score": score_out,
            "analysis": analysis_out,
            "calibration": calib_out,
        }
        self.manifest_manager.finish_run(run_id, result)
        return result
