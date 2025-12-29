# stephanie/memory/divergence_store.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

from sqlalchemy import and_

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.model_divergence import ModelDivergenceORM


class ModelDivergenceStore(BaseSQLAlchemyStore):
    """
    Store for ModelDivergenceORM with helpers to compute drift statistics
    per logical model.

    This is intentionally 'dumb': it does aggregates only.
    Policy about when to mark stale/degraded lives in ModelRegistryStore.
    """

    orm_model = ModelDivergenceORM
    default_order_by = ModelDivergenceORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "divergence"

    # --- core API ---

    def compute_window_stats_for_model(
        self,
        logical_model_id: int,
        *,
        window_hours: int = 24,
    ) -> Dict[str, Optional[float]]:
        """
        Compute basic drift statistics for a logical model over a recent time window.

        Returns a dict with:
          - count
          - mean_delta
          - mean_abs_delta
          - sign_flip_ratio
          - outlier_ratio
          - drift_score    (here just mean_abs_delta; can be replaced later)
        """
        def op(s):
            cutoff = datetime.utcnow() - timedelta(hours=window_hours)

            rows = (
                s.query(ModelDivergenceORM)
                .filter(
                    and_(
                        ModelDivergenceORM.logical_model_id == logical_model_id,
                        ModelDivergenceORM.created_at >= cutoff,
                    )
                )
                .all()
            )

            n = len(rows)
            if n == 0:
                return {
                    "count": 0,
                    "mean_delta": None,
                    "mean_abs_delta": None,
                    "sign_flip_ratio": None,
                    "outlier_ratio": None,
                    "drift_score": None,
                }

            sum_delta = 0.0
            sum_abs_delta = 0.0
            sign_flips = 0
            outliers = 0

            for r in rows:
                d = float(r.delta)
                ad = float(r.abs_delta)
                sum_delta += d
                sum_abs_delta += ad
                if r.sign_flip:
                    sign_flips += 1
                if r.is_outlier:
                    outliers += 1

            mean_delta = sum_delta / n
            mean_abs_delta = sum_abs_delta / n
            sign_flip_ratio = sign_flips / n
            outlier_ratio = outliers / n

            # For now, use mean_abs_delta as 'drift_score'.
            # You can later plug in EWMA, z-scores, or something fancier.
            drift_score = mean_abs_delta

            return {
                "count": n,
                "mean_delta": mean_delta,
                "mean_abs_delta": mean_abs_delta,
                "sign_flip_ratio": sign_flip_ratio,
                "outlier_ratio": outlier_ratio,
                "drift_score": drift_score,
            }

        stats = self._run(op)
        if self.logger:
            self.logger.debug(
                f"[DivergenceStore] Stats for model_id={logical_model_id} "
                f"over last {window_hours}h: {stats}"
            )
        return stats

    def compute_drift_score_for_model(
        self,
        logical_model_id: int,
        *,
        window_hours: int = 24,
    ) -> Optional[float]:
        """
        Convenience: just the drift_score.
        """
        stats = self.compute_window_stats_for_model(
            logical_model_id, window_hours=window_hours
        )
        return stats["drift_score"]

    def compute_outlier_ratio_for_model(
        self,
        logical_model_id: int,
        *,
        window_hours: int = 24,
    ) -> Optional[float]:
        """
        Convenience: just the outlier_ratio.
        """
        stats = self.compute_window_stats_for_model(
            logical_model_id, window_hours=window_hours
        )
        return stats["outlier_ratio"]
