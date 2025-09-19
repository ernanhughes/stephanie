# stephanie/memory/execution_step_store.py
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import asc

from stephanie.data.plan_trace import ExecutionStep
from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.plan_trace import ExecutionStepORM


class ExecutionStepStore(BaseSQLAlchemyStore):
    """
    Store for managing ExecutionStepORM objects in the database.
    Provides methods to insert, retrieve, and query execution steps.
    """
    orm_model = ExecutionStepORM
    default_order_by = ExecutionStepORM.step_order

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "execution_steps"
        self.table_name = "execution_steps"

    def name(self) -> str:
        return self.name

    # -------------------
    # Insert
    # -------------------
    def add(self, step: ExecutionStep) -> int:
        """
        Adds a new ExecutionStep dataclass by converting it into an ORM row.
        Returns the database ID.
        """
        orm_step = ExecutionStepORM(
            step_id=step.step_id,
            plan_trace_id=step.plan_trace_id,
            pipeline_run_id=step.pipeline_run_id,
            step_order=step.step_order,
            description=step.description,
            output_text=step.output_text,
            meta=step.meta,
        )
        return self.insert(orm_step)

    def insert(self, step: ExecutionStepORM) -> int:
        """Insert a fully populated ORM object and return its ID."""
        def op(s):
            
                s.add(step)
                s.flush()
                return step.id

        step_id = self._run(op)

        if self.logger:
            self.logger.log("ExecutionStepStored", {
                "db_id": step_id,
                "plan_trace_id": step.plan_trace_id,
                "step_order": step.step_order,
                "step_id_str": step.step_id,
                "created_at": step.created_at.isoformat() if step.created_at else None,
            })
        return step_id

    # -------------------
    # Retrieval
    # -------------------
    def get_by_id(self, step_id: int) -> Optional[ExecutionStepORM]:
        return self._run(lambda: self._scope().get(ExecutionStepORM, step_id))

    def get_by_step_id_str(self, step_id_str: str) -> Optional[ExecutionStepORM]:
        """Retrieve by application-level step_id string (not DB id)."""
        def op(s):
            return (
                self._scope()
                .query(ExecutionStepORM)
                .filter_by(step_id=step_id_str)
                .first()
            )
        return self._run(op)

    def get_steps_by_trace_id(
        self, plan_trace_id: int, ordered: bool = True
    ) -> List[ExecutionStepORM]:
        def op(s):
            q = s.query(ExecutionStepORM).filter_by(plan_trace_id=plan_trace_id)
            if ordered:
                q = q.order_by(asc(ExecutionStepORM.step_order))
            return q.all()
        return self._run(op)

    def get_by_evaluation_id(self, evaluation_id: int) -> Optional[ExecutionStepORM]:
        return self._run(
            lambda s: s.query(ExecutionStepORM)
            .filter_by(evaluation_id=evaluation_id)
            .first()
        )

    def get_all(self) -> List[ExecutionStepORM]:
        return self._run(lambda s: s.query(ExecutionStepORM).all())
