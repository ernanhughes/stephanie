# stephanie/memory/execution_step_store.py
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import asc

from stephanie.data.plan_trace import ExecutionStep
from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.plan_trace import ExecutionStepORM


class ExecutionStepStore(BaseSQLAlchemyStore):
    """
    Store for managing ExecutionStepORM objects in the database.
    Provides methods to insert, update, and query execution steps.
    """

    orm_model = ExecutionStepORM
    default_order_by = "step_order"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "execution_steps"

    # -------------------
    # INSERT / UPDATE
    # -------------------
    def insert(self, step: ExecutionStep) -> ExecutionStepORM:
        """
        Insert a new ExecutionStep dataclass as an ORM row.
        """

        def op(s):
            orm_step = ExecutionStepORM(
                step_id=str(step.step_id),
                plan_trace_id=step.plan_trace_id,
                pipeline_run_id=step.pipeline_run_id,
                step_order=step.step_order,
                description=step.description,
                output_text=step.output_text,
                meta=step.meta or {},
            )
            s.add(orm_step)
            s.flush()
            if self.logger:
                self.logger.log("ExecutionStepInserted", orm_step.to_dict())
            return orm_step

        return self._run(op)

    def upsert(self, step: ExecutionStep) -> ExecutionStepORM:
        """
        Update or insert an ExecutionStep dataclass into the DB.
        """

        def op(s):
            existing = (
                s.query(ExecutionStepORM)
                .filter_by(step_id=str(step.step_id))
                .first()
            )
            if existing:
                existing.plan_trace_id = step.plan_trace_id
                existing.pipeline_run_id = step.pipeline_run_id
                existing.step_order = step.step_order
                existing.step_type = step.step_type
                existing.description = step.description
                existing.output_text = step.output_text
                existing.meta = {**(existing.meta or {}), **(step.meta or {})}
                action = "ExecutionStepUpdated"
            else:
                existing = ExecutionStepORM(
                    step_id=str(step.step_id),
                    plan_trace_id=step.plan_trace_id,
                    pipeline_run_id=step.pipeline_run_id,
                    step_order=step.step_order,
                    step_type=step.step_type,
                    description=step.description,
                    output_text=step.output_text,
                    meta=step.meta or {},
                )
                s.add(existing)
                action = "ExecutionStepInserted"

            s.flush()
            if self.logger:
                self.logger.log(action, existing.to_dict())
            return existing

        return self._run(op)

    # -------------------
    # RETRIEVAL (unchanged)
    # -------------------
    def get_by_id(self, step_id: int) -> Optional[ExecutionStepORM]:
        return self._run(lambda s: s.get(ExecutionStepORM, step_id))

    def get_by_step_id(self, step_id_str: str) -> Optional[ExecutionStepORM]:
        return self._run(
            lambda s: s.query(ExecutionStepORM)
            .filter_by(step_id=step_id_str)
            .first()
        )

    def get_by_trace(
        self, plan_trace_id: int, ordered: bool = True
    ) -> List[ExecutionStepORM]:
        def op(s):
            q = s.query(ExecutionStepORM).filter_by(
                plan_trace_id=plan_trace_id
            )
            if ordered:
                q = q.order_by(asc(ExecutionStepORM.step_order))
            return q.all()

        return self._run(op)

    def get_all(self, limit: int = 100) -> List[ExecutionStepORM]:
        return self._run(
            lambda s: s.query(ExecutionStepORM).limit(limit).all()
        )

    def get_by_evaluation(
        self, evaluation_id: int
    ) -> Optional[ExecutionStepORM]:
        return self._run(
            lambda s: s.query(ExecutionStepORM)
            .filter_by(evaluation_id=evaluation_id)
            .first()
        )
