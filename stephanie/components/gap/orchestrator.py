# stephanie/components/gap/orchestrator.py
from __future__ import annotations

from stephanie.components.gap.models import GapConfig
from stephanie.components.gap.io.storage import GapStorage
from stephanie.components.gap.io.manifest import ManifestManager
from stephanie.components.gap.processors.scoring import ScoringProcessor
from stephanie.components.gap.processors.analysis import AnalysisProcessor
from stephanie.components.gap.processors.calibration import CalibrationProcessor


from typing import Any, Callable, Dict, Optional

class GapAnalysisOrchestrator:
    def __init__(self, config: GapConfig, container, logger, memory=None):
        self.config = config
        self.container = container
        self.logger = logger
        self.memory = memory

        # Storage & manifest
        self.storage = GapStorage(self.config.base_dir)
        self.manifest_manager = ManifestManager(self.storage)

        # Processors
        self.scoring_processor = ScoringProcessor(self.config, container, logger)
        self.analysis_processor = AnalysisProcessor(self.config, container, logger)
        self.calibration_processor = CalibrationProcessor(self.config, container, logger)

        # Make storage available via container (best-effort)
        try:
            container.register("storage", self.storage)
        except Exception:
            pass

    # ---- progress helpers ---------------------------------------------------
    def _mk_progress_cb(self, stage: str, every: int = 50) -> Callable[[int, int, Dict[str, Any]], None]:
        """
        Returns a callback like: cb(done:int, total:int, extras:dict)
        Processors can call this to emit heartbeat logs.
        """
        counter = {"last": -1}
        def _cb(done: int, total: int, extras: Dict[str, Any] | None = None):
            # limit log spam
            if total <= 0: 
                return
            pct = int((done / total) * 100)
            if pct // max(1, every) != counter["last"] // max(1, every) or done == total:
                payload = {"stage": stage, "done": done, "total": total, "percent": pct}
                if extras: payload.update(extras)
                self.logger.log("GapProgress", payload)
                counter["last"] = pct
        return _cb

    # ---- the end-to-end run -------------------------------------------------
    async def execute_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Context-based entrypoint so your GapAgent can just pass the pipeline context.
        Expected keys:
          - pipeline_run_id (str)
          - dataset (str)
          - triples_source (optional path)  # if you already built it elsewhere
        """
        run_id = context.get("pipeline_run_id")
        dataset = context.get("dataset", "unknown")
        models = {"hrm": self.config.hrm_scorers[0], "tiny": self.config.tiny_scorers[0]}

        # 0) initialize folders + manifest
        self.manifest_manager.start_run(
            run_id=run_id,
            dataset=dataset,
            models=models,
            dimensions=self.config.dimensions,
        )

        # 0.1) build triples if not supplied
        triples_source = context.get("triples_source")
        if not triples_source:
            from stephanie.components.gap.io.data import DataRetriever
            retriever = DataRetriever(str(self.storage.base_dir), self.logger)
            idx = retriever.build_triples_index(
                run_id=run_id,
                dimensions=self.config.dimensions,
                memory=self.memory,
                policy=self.config.dedupe_policy,
                per_dim_cap=self.config.per_dim_cap,
                per_dim_limit=context.get("per_dim_limit"),  # or None
            )
            triples_source = idx.jsonl_path
            self.manifest_manager.patch(run_id, {
                "inputs": {
                    "triples_index": idx.jsonl_path,
                    "triples_index_parquet": idx.parquet_path,
                    "triples_head": idx.head_path,
                    "triples_rows": idx.rows,
                    "per_dim_counts": idx.per_dim_counts,
                }
            })

        # 1) SCORING
        scoring_progress = self._mk_progress_cb("scoring", every=max(1, self.config.progress_log_every))
        scoring_out = await self.scoring_processor.score_all_triples(
            run_id=run_id,
            dataset=dataset,
            models=models,
            memory=self.memory,
            triples_source=triples_source,
            progress_cb=scoring_progress,
        )
        self.manifest_manager.patch(run_id, {"scoring": scoring_out})

        # 2) ANALYSIS
        analysis_progress = self._mk_progress_cb("analysis", every=10)
        analysis_out = await self.analysis_processor.perform_analysis(
            run_id=run_id,
            hrm_matrix_path=scoring_out["hrm_matrix_path"],
            tiny_matrix_path=scoring_out["tiny_matrix_path"],
            metric_names_path=scoring_out["metric_names_path"],
            rows_for_df_path=scoring_out.get("rows_for_df_path"),
            progress_cb=analysis_progress,
        )
        self.manifest_manager.patch(run_id, {"analysis": analysis_out})

        # 3) CALIBRATION
        calibration_progress = self._mk_progress_cb("calibration", every=20)
        calibration_out = await self.calibration_processor.analyze_calibration(
            run_id=run_id,
            hrm_matrix_path=scoring_out["hrm_matrix_path"],
            tiny_matrix_path=scoring_out["tiny_matrix_path"],
            metric_names_path=scoring_out["metric_names_path"],
            progress_cb=calibration_progress,
        )
        self.manifest_manager.patch(run_id, {"calibration": calibration_out})

        headline = {
            "run_id": run_id,
            "paths": {
                "hrm_timeline_gif": scoring_out.get("hrm_timeline_gif"),
                "tiny_timeline_gif": scoring_out.get("tiny_timeline_gif"),
                "frontier_png": analysis_out.get("frontier_png"),
                "delta_abs_heat_png": analysis_out.get("delta_abs_heat_png"),
                "intermodel_delta_json": analysis_out.get("intermodel_delta_json"),
                "intensity_report_json": analysis_out.get("intensity_report_json"),
                "hrm_vpm_phos_png": analysis_out.get("hrm_vpm_phos_png"),
                "tiny_vpm_phos_png": analysis_out.get("tiny_vpm_phos_png"),
                "calibration_params_json": calibration_out.get("calibration_params_json"),
                "routing_summary_json": calibration_out.get("routing_summary_json"),
                "routing_detail_json": calibration_out.get("routing_detail_json"),
            },
            "metrics": {
                "delta_mass": analysis_out.get("delta_mass"),
                "overlap": analysis_out.get("overlap"),
                "hrm_vs_tiny_usage_rate": calibration_out.get("usage_rate"),
                "avg_mae_vs_hrm": calibration_out.get("avg_mae_vs_hrm"),
            },
        }
        self.manifest_manager.patch(run_id, {"headline": headline})
        self.logger.log("GapRunComplete", {"run_id": run_id})
        return headline
