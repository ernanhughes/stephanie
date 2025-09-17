# stephanie/memory/mars_result_store.py
from __future__ import annotations

from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.mars_result import MARSResultORM


class MARSResultStore(BaseSQLAlchemyStore):
    orm_model = MARSResultORM
    default_order_by = MARSResultORM.created_at.desc()
    
    def __init__(self, session: Session, logger=None):
        super().__init__(session, logger)
        self.name = "mars_results"

    def name(self) -> str:
        return self.name

    def add(self, pipeline_run_id, plan_trace_id, result: dict):
        orm = MARSResultORM(
            pipeline_run_id=pipeline_run_id,
            plan_trace_id=plan_trace_id,
            **result
        )
        self.session.add(orm)
        self.session.commit()
        if self.logger:
            self.logger.log("MARSResultStored", {"id": orm.id, "dimension": orm.dimension})
        return orm

    def get_by_trace(self, trace_id: str):
        return self.session.query(MARSResultORM).filter_by(plan_trace_id=trace_id).all()

    def get_by_run_id(self, run_id: int):
        return (
            self.session.query(MARSResultORM)
            .filter(MARSResultORM.pipeline_run_id == run_id)
            .all()
        )

    def get_by_plan_trace(self, trace_id: str):
        return (
            self.session.query(MARSResultORM)
            .filter(MARSResultORM.plan_trace_id == trace_id)
            .all()
        )

    def get_recent(self, limit: int = 50):
        return (
            self.session.query(MARSResultORM)
            .order_by(MARSResultORM.created_at.desc())
            .limit(limit)
            .all()
        )
