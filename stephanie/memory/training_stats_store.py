# stephanie/memory/training_stats_store.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Query

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.training_stats import TrainingStatsORM


class TrainingStatsStore(BaseSQLAlchemyStore):
    """
    Store for recording and querying training runs (TrainingStatsORM).
    """
    orm_model = TrainingStatsORM
    default_order_by = TrainingStatsORM.start_time.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "training_stats"

    # -------------------
    # Create / Update
    # -------------------
    def add(self, stats_data: Dict[str, Any]) -> TrainingStatsORM:
        """
        Insert a single TrainingStats row. Accepts a plain dict of fields.
        """
        def op(s):
            row = TrainingStatsORM(**stats_data)
            s.add(row)
            s.flush()
            if self.logger:
                self.logger.log("TrainingStatsInserted", {"id": row.id, "model_type": row.model_type, "dimension": row.dimension})
            return row
        return self._run(op)

    def add_from_result(
        self,
        stats: Dict[str, Any],
        *,
        model_type: str,
        target_type: str,
        dimension: str,
        version: str,
        embedding_type: str,
        config: Optional[Dict[str, Any]] = None,
        sample_count: int = 0,
        valid_samples: int = 0,
        invalid_samples: int = 0,
        goal_id: Optional[int] = None,
        model_version_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        pipeline_run_id: Optional[int] = None,
    ) -> TrainingStatsORM:
        """
        Convenience wrapper that mirrors your .from_dict() shape and adds IDs/links.
        """
        # __dict__ includes SQLAlchemy internals; build a clean dict explicitly:
        clean = dict(
            model_type=model_type,
            target_type=target_type,
            dimension=dimension,
            version=version,
            embedding_type=embedding_type,
            q_loss=stats.get("q_loss"),
            v_loss=stats.get("v_loss"),
            pi_loss=stats.get("pi_loss"),
            avg_q_loss=stats.get("avg_q_loss"),
            avg_v_loss=stats.get("avg_v_loss"),
            avg_pi_loss=stats.get("avg_pi_loss"),
            policy_entropy=stats.get("policy_entropy"),
            policy_stability=stats.get("policy_stability"),
            policy_logits=stats.get("policy_logits"),
            config=config or {},
            sample_count=int(sample_count or 0),
            valid_samples=int(valid_samples or 0),
            invalid_samples=int(invalid_samples or 0),
            start_time=start_time or datetime.now(),
            end_time=end_time,
            goal_id=goal_id,
            model_version_id=model_version_id,
            pipeline_run_id=pipeline_run_id,
        )
        return self.add(clean)

    def mark_ended(
        self, stat_id: int, *, end_time: Optional[datetime] = None, extra_metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[TrainingStatsORM]:
        """
        Set end_time and optionally update metrics on a run.
        """
        def op(s):
            row = s.get(TrainingStatsORM, stat_id)
            if not row:
                return None
            row.end_time = end_time or datetime.now()
            if extra_metrics:
                # Update only known metric fields if present
                for k in ("q_loss", "v_loss", "pi_loss", "avg_q_loss", "avg_v_loss", "avg_pi_loss",
                          "policy_entropy", "policy_stability"):
                    if k in extra_metrics:
                        setattr(row, k, extra_metrics[k])
            s.add(row)
            s.flush()
            if self.logger:
                self.logger.log("TrainingStatsEnded", {"id": row.id, "ended_at": row.end_time.isoformat()})
            return row
        return self._run(op)

    # -------------------
    # Retrieval
    # -------------------
    def get(self, stat_id: int) -> Optional[TrainingStatsORM]:
        def op(s):
            return s.get(TrainingStatsORM, stat_id)
        return self._run(op)

    def list_recent(self, *, limit: int = 50) -> List[TrainingStatsORM]:
        def op(s):
            return (
                s.query(TrainingStatsORM)
                .order_by(self.default_order_by)
                .limit(int(limit))
                .all()
            )
        return self._run(op)

    def list_by_filters(
        self,
        *,
        model_type: Optional[str] = None,
        target_type: Optional[str] = None,
        dimension: Optional[str] = None,
        version: Optional[str] = None,
        embedding_type: Optional[str] = None,
        goal_id: Optional[int] = None,
        model_version_id: Optional[int] = None,
        limit: int = 200,
        order_desc: bool = True,
    ) -> List[TrainingStatsORM]:
        """
        Generic filtered listing; mirrors your schema axes.
        """
        def op(s):
            q: Query = s.query(TrainingStatsORM)
            if model_type:
                q = q.filter(TrainingStatsORM.model_type == model_type)
            if target_type:
                q = q.filter(TrainingStatsORM.target_type == target_type)
            if dimension:
                q = q.filter(TrainingStatsORM.dimension == dimension)
            if version:
                q = q.filter(TrainingStatsORM.version == version)
            if embedding_type:
                q = q.filter(TrainingStatsORM.embedding_type == embedding_type)
            if goal_id is not None:
                q = q.filter(TrainingStatsORM.goal_id == goal_id)
            if model_version_id is not None:
                q = q.filter(TrainingStatsORM.model_version_id == model_version_id)

            q = q.order_by(TrainingStatsORM.start_time.desc() if order_desc else TrainingStatsORM.start_time.asc())
            return q.limit(int(limit)).all()
        return self._run(op)

    def latest_for(
        self,
        *,
        model_type: str,
        target_type: str,
        dimension: str,
        embedding_type: Optional[str] = None,
    ) -> Optional[TrainingStatsORM]:
        """
        Get most recent run matching the axes you care about.
        """
        def op(s):
            q = (
                s.query(TrainingStatsORM)
                .filter(
                    TrainingStatsORM.model_type == model_type,
                    TrainingStatsORM.target_type == target_type,
                    TrainingStatsORM.dimension == dimension,
                )
            )
            if embedding_type:
                q = q.filter(TrainingStatsORM.embedding_type == embedding_type)
            return q.order_by(TrainingStatsORM.start_time.desc()).first()
        return self._run(op)

    def best_by_metric(
        self,
        *,
        model_type: str,
        target_type: str,
        dimension: str,
        metric: str = "avg_q_loss",
        minimize: bool = True,
        limit: int = 1,
    ) -> List[TrainingStatsORM]:
        """
        Return top runs by a numeric metric (default: lowest avg_q_loss).
        """
        assert metric in {"q_loss", "v_loss", "pi_loss", "avg_q_loss", "avg_v_loss", "avg_pi_loss"}
        def op(s):
            col = getattr(TrainingStatsORM, metric)
            order = col.asc() if minimize else col.desc()
            return (
                s.query(TrainingStatsORM)
                .filter(
                    TrainingStatsORM.model_type == model_type,
                    TrainingStatsORM.target_type == target_type,
                    TrainingStatsORM.dimension == dimension,
                    col.isnot(None),
                )
                .order_by(order, TrainingStatsORM.start_time.desc())
                .limit(int(limit))
                .all()
            )
        return self._run(op)

    def aggregate_summary(
        self,
        *,
        model_type: Optional[str] = None,
        target_type: Optional[str] = None,
        dimension: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Lightweight summary: counts and average losses across (optional) filters.
        """
        def op(s):
            q = s.query(
                func.count(TrainingStatsORM.id),
                func.avg(TrainingStatsORM.avg_q_loss),
                func.avg(TrainingStatsORM.avg_v_loss),
                func.avg(TrainingStatsORM.avg_pi_loss),
                func.avg(TrainingStatsORM.sample_count),
            )
            if model_type:
                q = q.filter(TrainingStatsORM.model_type == model_type)
            if target_type:
                q = q.filter(TrainingStatsORM.target_type == target_type)
            if dimension:
                q = q.filter(TrainingStatsORM.dimension == dimension)
            if version:
                q = q.filter(TrainingStatsORM.version == version)

            total, avg_q, avg_v, avg_pi, avg_samples = q.one()
            return {
                "total_runs": int(total or 0),
                "avg_avg_q_loss": float(avg_q) if avg_q is not None else None,
                "avg_avg_v_loss": float(avg_v) if avg_v is not None else None,
                "avg_avg_pi_loss": float(avg_pi) if avg_pi is not None else None,
                "avg_sample_count": float(avg_samples) if avg_samples is not None else None,
            }
        return self._run(op)
