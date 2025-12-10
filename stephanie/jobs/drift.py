# stephanie/services/drift_job.py
from __future__ import annotations

from typing import Optional

from stephanie.memory.model_registry_store import ModelRegistryStore
from stephanie.memory.model_divergence_store import ModelDivergenceStore
from stephanie.models.model import ModelORM


class DriftDetectionJob:
    """
    Periodic job that:
      1) Aggregates divergences into drift stats
      2) Updates ModelHealthORM via ModelRegistryStore
      3) Lets ModelRegistryStore derive ModelORM.status from health
    """

    def __init__(
        self,
        model_registry: ModelRegistryStore,
        divergence_store: ModelDivergenceStore,
        logger=None,
        window_hours: int = 24,
    ):
        self.model_registry = model_registry
        self.divergence_store = divergence_store
        self.window_hours = window_hours
        self.logger = logger

    def run_once(self):
        models = self.model_registry.list_models(limit=10_000)  # or some sensible limit

        for m in models:
            self._process_model(m)

    def _process_model(self, model: ModelORM):
        stats = self.divergence_store.compute_window_stats_for_model(
            model.id,
            window_hours=self.window_hours,
        )

        # No data yet → skip health update, but we might still default status later
        if stats["count"] == 0:
            if self.logger:
                self.logger.debug(
                    f"[DriftDetectionJob] No divergences for model_id={model.id} "
                    f"({model.model_type}/{model.target_type}/{model.dimension}); skipping"
                )
            return

        drift_score = stats["drift_score"]
        outlier_ratio = stats["outlier_ratio"]

        # Update health (you can add more fields as you start computing them)
        self.model_registry.upsert_health(
            model,
            drift_score=drift_score,
            mean_delta=stats["mean_delta"],
            mean_abs_delta=stats["mean_abs_delta"],
            sign_flip_ratio=stats["sign_flip_ratio"],
            outlier_ratio=outlier_ratio,
            # num_measurements could come from MeasurementQuality or a join later
            metrics={"window_hours": self.window_hours},
        )

        # Now automatically derive lifecycle status from health
        updated_model = self.model_registry.update_status_from_health(model.id)

        if self.logger:
            self.logger.info(
                f"[DriftDetectionJob] Updated model_id={model.id} "
                f"status={updated_model.status} drift={drift_score} "
                f"outliers={outlier_ratio}"
            )
