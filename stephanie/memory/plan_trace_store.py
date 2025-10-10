# stephanie/memory/plan_trace_store.py
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import joinedload

from stephanie.data.plan_trace import PlanTrace
from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.goal import GoalORM
from stephanie.models.plan_trace import ExecutionStepORM, PlanTraceORM
from stephanie.models.plan_trace_reuse_link import PlanTraceReuseLinkORM
from stephanie.models.plan_trace_revision import PlanTraceRevisionORM


class PlanTraceStore(BaseSQLAlchemyStore):
    orm_model = PlanTraceORM
    default_order_by = "created_at"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "plan_traces"

    # --------------------
    # INSERT / UPDATE
    # --------------------
    def insert(self, plan_trace: PlanTrace) -> PlanTraceORM:
        """
        Insert a new PlanTrace and its steps.
        """
        def op(s):
            orm_trace = PlanTraceORM(
                trace_id=plan_trace.trace_id,
                pipeline_run_id=plan_trace.pipeline_run_id,
                goal_id=plan_trace.goal_id,
                plan_signature=plan_trace.plan_signature,
                final_output_text=plan_trace.final_output_text,
                target_epistemic_quality=plan_trace.target_epistemic_quality,
                target_epistemic_quality_source=plan_trace.target_epistemic_quality_source,
                meta=plan_trace.meta,
            )
            for i, step in enumerate(plan_trace.execution_steps, 1):
                orm_trace.execution_steps.append(
                    ExecutionStepORM(
                        plan_trace=orm_trace,
                        pipeline_run_id=plan_trace.pipeline_run_id,
                        step_order=step.step_order or i,
                        step_id=str(step.step_id),
                        description=step.description,
                        output_text=step.output_text,
                        output_embedding_id=None,
                        meta={**(step.attributes or {}), **(step.meta or {})},
                    )
                )
            s.add(orm_trace)
            s.flush()
            if self.logger:
                self.logger.log("PlanTraceInserted", orm_trace.to_dict())
            return orm_trace

        return self._run(op)

    def upsert(self, plan_trace: PlanTrace) -> PlanTraceORM:
        """
        Update or insert a PlanTrace (and its steps).
        """
        def op(s):
            existing = (
                s.query(PlanTraceORM)
                .filter_by(trace_id=plan_trace.trace_id)
                .options(joinedload(PlanTraceORM.execution_steps))
                .first()
            )
            if existing:
                existing.final_output_text = plan_trace.final_output_text
                existing.target_epistemic_quality = plan_trace.target_epistemic_quality
                existing.target_epistemic_quality_source = plan_trace.target_epistemic_quality_source
                if isinstance(plan_trace.meta, dict):
                    existing.meta = {**(existing.meta or {}), **plan_trace.meta}
                existing.execution_steps.clear()
                for i, step in enumerate(plan_trace.execution_steps, 1):
                    existing.execution_steps.append(
                        ExecutionStepORM(
                            plan_trace=existing,
                            pipeline_run_id=plan_trace.pipeline_run_id,
                            step_order=step.step_order or i,
                            step_id=str(step.step_id),
                            description=step.description,
                            output_text=step.output_text,
                            meta={**(step.attributes or {}), **(step.meta or {})},
                        )
                    )
                action = "PlanTraceUpdated"
            else:
                existing = PlanTraceORM(
                    trace_id=plan_trace.trace_id,
                    pipeline_run_id=plan_trace.pipeline_run_id,
                    goal_id=plan_trace.goal_id,
                    plan_signature=plan_trace.plan_signature,
                    final_output_text=plan_trace.final_output_text,
                    target_epistemic_quality=plan_trace.target_epistemic_quality,
                    target_epistemic_quality_source=plan_trace.target_epistemic_quality_source,
                    meta=plan_trace.meta,
                )
                s.add(existing)
                action = "PlanTraceInserted"

            s.flush()
            if self.logger:
                self.logger.log(action, existing.to_dict())
            return existing

        return self._run(op)

    # --------------------
    # RETRIEVAL
    # --------------------
    def get_by_id(self, trace_id: int) -> Optional[PlanTraceORM]:
        return self._run(lambda s: s.get(PlanTraceORM, trace_id))

    def get_by_trace_id(self, trace_id: str) -> Optional[PlanTraceORM]:
        return self._run(
            lambda s: s.query(PlanTraceORM)
            .filter_by(trace_id=trace_id)
            .options(joinedload(PlanTraceORM.goal), joinedload(PlanTraceORM.execution_steps))
            .first()
        )

    def get_by_run_id(self, run_id: str) -> Optional[PlanTraceORM]:
        return self._run(
            lambda s: s.query(PlanTraceORM)
            .filter_by(pipeline_run_id=run_id)
            .options(joinedload(PlanTraceORM.goal))
            .first()
        )

    def get_by_goal_id(self, goal_id: int) -> List[PlanTraceORM]:
        return self._run(
            lambda s: s.query(PlanTraceORM)
            .filter_by(goal_id=goal_id)
            .options(joinedload(PlanTraceORM.goal))
            .all()
        )

    def get_recent(self, limit: int = 10) -> List[PlanTraceORM]:
        return self._run(
            lambda s: s.query(PlanTraceORM)
            .order_by(desc(PlanTraceORM.created_at))
            .limit(limit)
            .all()
        )

    def get_all(self, limit: int = 100) -> List[PlanTraceORM]:
        return self._run(
            lambda s: s.query(PlanTraceORM)
            .order_by(desc(PlanTraceORM.created_at))
            .limit(limit)
            .all()
        )

    def get_goal_text(self, trace_id: str) -> Optional[str]:
        return self._run(
            lambda s: (
                s.query(GoalORM.goal_text)
                .join(PlanTraceORM, GoalORM.id == PlanTraceORM.goal_id)
                .filter(PlanTraceORM.trace_id == trace_id)
                .first()
            )[0]
            if s.query(GoalORM.goal_text)
            .join(PlanTraceORM, GoalORM.id == PlanTraceORM.goal_id)
            .filter(PlanTraceORM.trace_id == trace_id)
            .first()
            else None
        )

    # --------------------
    # REUSE LINKS / REVISIONS
    # --------------------
    def add_reuse_link(self, parent_trace_id: str, child_trace_id: str):
        def op(s):
            if parent_trace_id == child_trace_id:
                if self.logger:
                    self.logger.log(
                        "PlanTraceReuseLinkSkipped",
                        {"reason": "parent == child", "trace_id": parent_trace_id},
                    )
                return None
            link = PlanTraceReuseLinkORM(
                parent_trace_id=parent_trace_id, child_trace_id=child_trace_id
            )
            s.add(link)
            s.flush()
            return link.id

        return self._run(op)

    def get_reuse_links_for_trace(self, trace_id: str):
        return self._run(
            lambda s: s.query(PlanTraceReuseLinkORM)
            .filter(
                (PlanTraceReuseLinkORM.parent_trace_id == trace_id)
                | (PlanTraceReuseLinkORM.child_trace_id == trace_id)
            )
            .all()
        )

    def add_revision(self, trace_id: str, revision_type: str, revision_text: str, source: str = "user"):
        def op(s):
            rev = PlanTraceRevisionORM(
                plan_trace_id=trace_id,
                revision_type=revision_type,
                revision_text=revision_text,
                source=source,
            )
            s.add(rev)
            s.flush()
            return rev

        return self._run(op)

    def get_revisions(self, trace_id: str) -> list[PlanTraceRevisionORM]:
        return self._run(
            lambda s: s.query(PlanTraceRevisionORM)
            .filter_by(plan_trace_id=trace_id)
            .order_by(PlanTraceRevisionORM.created_at)
            .all()
        )

    # --------------------
    # EXECUTION STEPS
    # --------------------
    def get_step_by_id(self, step_id: int) -> Optional[ExecutionStepORM]:
        return self._run(lambda s: s.get(ExecutionStepORM, step_id))
