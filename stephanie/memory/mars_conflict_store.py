from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.mars_conflict import MARSConflictORM


class MARSConflictStore(BaseSQLAlchemyStore):
    orm_model = MARSConflictORM
    default_order_by = MARSConflictORM.created_at.desc()
    
    def __init__(self, session: Session, logger=None):
        super().__init__(session, logger)
        self.name = "mars_conflicts"

    def name(self) -> str:
        return self.name

    def add(self, pipeline_run_id, plan_trace_id, dimension, conflict, delta, explanation, agreement_score=None, preferred_model=None):
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
        self.session.add(orm)
        self.session.commit()
        if self.logger:
            self.logger.log("MARSConflictStored", {
                "id": orm.id,
                "dimension": orm.dimension,
                "conflict": orm.primary_conflict,
                "delta": orm.delta,
            })
        return orm

    def get_by_trace(self, trace_id: str):
        return self.session.query(MARSConflictORM).filter_by(plan_trace_id=trace_id).all()

    def get_by_run_id(self, run_id: int):
        return self.session.query(MARSConflictORM).filter_by(pipeline_run_id=run_id).all()

    def get_recent(self, limit: int = 50):
        return (
            self.session.query(MARSConflictORM)
            .order_by(MARSConflictORM.created_at.desc())
            .limit(limit)
            .all()
        )
