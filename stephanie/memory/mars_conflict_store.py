# stephanie/memory/mars_conflict_store.py
from __future__ import annotations

from typing import List

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.mars_conflict import MARSConflictORM


class MARSConflictStore(BaseSQLAlchemyStore):
    orm_model = MARSConflictORM
    default_order_by = MARSConflictORM.created_at.desc()
    
    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "mars_conflicts"

    def name(self) -> str:
        return self.name

    # -------------------
    # Insert
    # -------------------
    def add(
        self,
        pipeline_run_id: int,
        plan_trace_id: int,
        dimension: str,
        conflict: str,
        delta: float,
        explanation: str,
        agreement_score: float | None = None,
        preferred_model: str | None = None,
    ) -> MARSConflictORM:
        """Insert a new MARS conflict row."""
        def op():
            s = self._scope()
            orm = MARSConflictORM(
                pipeline_run_id=pipeline_run_id,
                plan_trace_id=plan_trace_id,
                dimension=dimension,
                primary_conflict=conflict,
                delta=delta,
                explanation=explanation,
                agreement_score=agreement_score,
                preferred_model=preferred_model,
            )
            s.add(orm)
            s.flush()
            if self.logger:
                self.logger.log(
                    "MARSConflictStored",
                    {
                        "id": orm.id,
                        "dimension": orm.dimension,
                        "conflict": orm.primary_conflict,
                        "delta": orm.delta,
                    },
                )
            return orm
        return self._run(op)

    # -------------------
    # Retrieval
    # -------------------
    def get_by_trace(self, trace_id: int) -> List[MARSConflictORM]:
        def op():
            return (
                self._scope().query(MARSConflictORM)
                .filter_by(plan_trace_id=trace_id)
                .all()
            )
        return self._run(op)

    def get_by_run_id(self, run_id: int) -> List[MARSConflictORM]:
        def op():
            return (
                self._scope().query(MARSConflictORM)
                .filter_by(pipeline_run_id=run_id)
                .all()
            )
        return self._run(op)

    def get_recent(self, limit: int = 50) -> List[MARSConflictORM]:
        def op():
            return (
                self._scope().query(MARSConflictORM)
                .order_by(MARSConflictORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)
