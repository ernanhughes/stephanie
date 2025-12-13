# stephanie/memory/model_registry_store.py
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy import and_

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.model import ModelHealthORM, ModelORM


class ModelRegistryStore(BaseSQLAlchemyStore):
    """
    Registry for logical models (sicql/hrm/tiny/etc.) and their health.

    This does NOT replace ModelArtifactORM or ModelVersionORM – it sits above
    them and lets your Supervisor/Locator reason about:
      - what's registered,
      - which version is active,
      - how healthy it is.
    """

    orm_model = ModelORM
    default_order_by = ModelORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "model_registry"

    # --- core helpers ---

    def _get_model_row(
        self,
        model_type: str,
        target_type: str,
        dimension: str,
        score_mode: Optional[str] = None,
    ) -> Optional[ModelORM]:
        def op(s):
            q = s.query(ModelORM).filter(
                and_(
                    ModelORM.model_type == model_type,
                    ModelORM.target_type == target_type,
                    ModelORM.dimension == dimension,
                )
            )
            if score_mode is not None:
                q = q.filter(ModelORM.score_mode == score_mode)
            else:
                q = q.filter(ModelORM.score_mode.is_(None))
            return q.one_or_none()
        return self._run(op)

    # --- API ---

    def register_model(
        self,
        *,
        model_type: str,
        target_type: str,
        dimension: str,
        score_mode: Optional[str] = None,
        description: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> ModelORM:
        """
        Get or create a logical model entry.
        """
        def op(s):
            q = s.query(ModelORM).filter(
                and_(
                    ModelORM.model_type == model_type,
                    ModelORM.target_type == target_type,
                    ModelORM.dimension == dimension,
                )
            )
            if score_mode is not None:
                q = q.filter(ModelORM.score_mode == score_mode)
            else:
                q = q.filter(ModelORM.score_mode.is_(None))

            row = q.one_or_none()
            if row:
                return row

            now = datetime.now()
            row = ModelORM(
                model_type=model_type,
                target_type=target_type,
                dimension=dimension,
                score_mode=score_mode,
                description=description,
                meta=meta or {},
                status="unknown",
                created_at=now,
                updated_at=now,
            )
            s.add(row)
            s.flush()
            return row

        row = self._run(op)
        if self.logger:
            self.logger.info(
                f"[ModelRegistryStore] Registered model "
                f"{row.model_type}/{row.target_type}/{row.dimension}"
                f"{'/' + row.score_mode if row.score_mode else ''} (id={row.id})"
            )
        return row

    def set_active_version(
        self,
        *,
        model_type: str,
        target_type: str,
        dimension: str,
        score_mode: Optional[str],
        active_version: Optional[str],
        status: Optional[str] = None,
    ) -> Optional[ModelORM]:
        """
        Update active_version (string) for a logical model.
        Optionally update its lifecycle status.
        """
        def op(s):
            q = s.query(ModelORM).filter(
                and_(
                    ModelORM.model_type == model_type,
                    ModelORM.target_type == target_type,
                    ModelORM.dimension == dimension,
                )
            )
            if score_mode is not None:
                q = q.filter(ModelORM.score_mode == score_mode)
            else:
                q = q.filter(ModelORM.score_mode.is_(None))

            row = q.one_or_none()
            if not row:
                return None

            row.active_version = active_version
            if status is not None:
                row.status = status
            row.updated_at = datetime.now()
            s.add(row)
            s.flush()
            return row

        row = self._run(op)
        if row and self.logger:
            self.logger.info(
                f"[ModelRegistryStore] Set active version for "
                f"{row.model_type}/{row.target_type}/{row.dimension}"
                f"{'/' + row.score_mode if row.score_mode else ''}: "
                f"{row.active_version} (status={row.status})"
            )
        return row

    def update_status(
        self,
        *,
        model_type: str,
        target_type: str,
        dimension: str,
        score_mode: Optional[str],
        status: str,
    ) -> Optional[ModelORM]:
        """
        Update lifecycle status only (e.g., 'training', 'stale', 'ready').
        """
        return self.set_active_version(
            model_type=model_type,
            target_type=target_type,
            dimension=dimension,
            score_mode=score_mode,
            active_version=None,  # keep unchanged
            status=status,
        )

    def list_models(self, limit: int = 100) -> List[ModelORM]:
        def op(s):
            return (
                s.query(ModelORM)
                .order_by(
                    ModelORM.model_type.asc(),
                    ModelORM.target_type.asc(),
                    ModelORM.dimension.asc(),
                    ModelORM.score_mode.asc().nullsfirst(),
                )
                .limit(limit)
                .all()
            )
        return self._run(op)

    # --- health API ---

    def get_health(self, model_id: int) -> Optional[ModelHealthORM]:
        def op(s):
            return (
                s.query(ModelHealthORM)
                .filter(ModelHealthORM.model_id == model_id)
                .one_or_none()
            )
        return self._run(op)

    def upsert_health(
        self,
        model: ModelORM,
        *,
        health_status: Optional[str] = None,
        drift_score: Optional[float] = None,
        mean_delta: Optional[float] = None,
        mean_abs_delta: Optional[float] = None,
        sign_flip_ratio: Optional[float] = None,
        outlier_ratio: Optional[float] = None,
        num_measurements: Optional[int] = None,
        num_training_examples: Optional[int] = None,
        data_freshness_days: Optional[float] = None,
        last_retrain_at: Optional[datetime] = None,
        metrics: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> ModelHealthORM:
        """
        Create or update the health row for a given logical model.
        """
        def op(s):
            h = (
                s.query(ModelHealthORM)
                .filter(ModelHealthORM.model_id == model.id)
                .one_or_none()
            )
            now = datetime.now()

            if not h:
                h = ModelHealthORM(
                    model_id=model.id,
                    created_at=now,
                )
                s.add(h)

            if health_status is not None:
                h.health_status = health_status
            if drift_score is not None:
                h.drift_score = drift_score
            if mean_delta is not None:
                h.mean_delta = mean_delta
            if mean_abs_delta is not None:
                h.mean_abs_delta = mean_abs_delta
            if sign_flip_ratio is not None:
                h.sign_flip_ratio = sign_flip_ratio
            if outlier_ratio is not None:
                h.outlier_ratio = outlier_ratio
            if num_measurements is not None:
                h.num_measurements = num_measurements
            if num_training_examples is not None:
                h.num_training_examples = num_training_examples
            if data_freshness_days is not None:
                h.data_freshness_days = data_freshness_days
            if last_retrain_at is not None:
                h.last_retrain_at = last_retrain_at
            if metrics is not None:
                h.metrics = metrics
            if notes is not None:
                h.notes = notes

            h.last_evaluated_at = now
            h.updated_at = now
            s.flush()
            return h

        h = self._run(op)
        if self.logger:
            self.logger.info(
                f"[ModelRegistryStore] Updated health for model_id={model.id} "
                f"({model.model_type}/{model.target_type}/{model.dimension}"
                f"{'/' + model.score_mode if model.score_mode else ''}): "
                f"status={h.health_status}, drift={h.drift_score}"
            )
        return h

    def get_model_with_health(
        self,
        *,
        model_type: str,
        target_type: str,
        dimension: str,
        score_mode: Optional[str] = None,
    ) -> Optional[Tuple[ModelORM, Optional[ModelHealthORM]]]:
        """
        Convenience method for the Supervisor / Locator: fetch logical model + health.
        """
        def op(s):
            q = s.query(ModelORM).filter(
                and_(
                    ModelORM.model_type == model_type,
                    ModelORM.target_type == target_type,
                    ModelORM.dimension == dimension,
                )
            )
            if score_mode is not None:
                q = q.filter(ModelORM.score_mode == score_mode)
            else:
                q = q.filter(ModelORM.score_mode.is_(None))

            m = q.one_or_none()
            if not m:
                return None

            h = (
                s.query(ModelHealthORM)
                .filter(ModelHealthORM.model_id == m.id)
                .one_or_none()
            )
            return (m, h)

        return self._run(op)


    def update_status_from_health(self, model_id: int) -> Optional[ModelORM]:
        """
        Automatically update ModelORM.status based on ModelHealthORM metrics.

        Returns the updated ModelORM (or None if model/health missing).
        """
        def op(s):
            m = s.query(ModelORM).get(model_id)
            if not m:
                return None

            h = (
                s.query(ModelHealthORM)
                .filter(ModelHealthORM.model_id == model_id)
                .one_or_none()
            )
            if not h:
                return m  # nothing to update from

            # Safe defaults if metrics are None
            drift = h.drift_score or 0.0
            outliers = h.outlier_ratio or 0.0

            new_status = m.status

            # Thresholds should eventually come from config; hardcode for now
            if outliers > 0.3 or drift > 5.0:
                new_status = "stale"
            elif outliers > 0.1 or drift > 2.0:
                new_status = "degraded"
            else:
                new_status = "ready"

            if new_status != m.status:
                m.status = new_status
                m.updated_at = datetime.utcnow()
                s.add(m)

            return m

        return self._run(op)


    def list_by_status(self, status: str, limit: int = 100) -> list[ModelORM]:
        def op(s):
            return (
                s.query(ModelORM)
                .filter(ModelORM.status == status)
                .order_by(ModelORM.updated_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)
