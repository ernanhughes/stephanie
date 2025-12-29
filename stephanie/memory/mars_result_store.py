# stephanie/memory/mars_result_store.py
from __future__ import annotations

from typing import List

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.mars_result import MARSResultORM


class MARSResultStore(BaseSQLAlchemyStore):
    orm_model = MARSResultORM
    default_order_by = MARSResultORM.created_at.desc()
    
    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "mars_results"

    # -------------------
    # Insert
    # -------------------
    def add(
        self,
        pipeline_run_id: int,
        plan_trace_id: int,
        result: dict,
    ) -> MARSResultORM:
        """Insert a new MARS result row."""
        def op(s):
            orm = MARSResultORM(
                pipeline_run_id=pipeline_run_id,
                plan_trace_id=plan_trace_id,
                **result,
            )
            s.add(orm)
            s.flush()
            if self.logger:
                self.logger.log(
                    "MARSResultStored",
                    {"id": orm.id, "dimension": getattr(orm, "dimension", None)},
                )
            return orm
        return self._run(op)

    # -------------------
    # Retrieval
    # -------------------
    def get_by_trace(self, trace_id: int) -> List[MARSResultORM]:
        def op(s):
            return (
                s.query(MARSResultORM)
                .filter_by(plan_trace_id=trace_id)
                .all()
            )
        return self._run(op)

    def get_by_run_id(self, run_id: int) -> List[MARSResultORM]:
        def op(s):
            return (
                s.query(MARSResultORM)
                .filter(MARSResultORM.pipeline_run_id == run_id)
                .all()
            )
        return self._run(op)

    def get_by_plan_trace(self, trace_id: int) -> List[MARSResultORM]:
        def op(s):
            return (
                s.query(MARSResultORM)
                .filter(MARSResultORM.plan_trace_id == trace_id)
                .all()
            )
        return self._run(op)

    def get_recent(self, limit: int = 50) -> List[MARSResultORM]:
        def op(s):
            return (
                s.query(MARSResultORM)
                .order_by(MARSResultORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)
