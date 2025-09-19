# stephanie/memory/plan_trace_store.py
from __future__ import annotations

import traceback
from typing import List, Optional
from sqlalchemy import desc

from stephanie.data.plan_trace import PlanTrace
from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.goal import GoalORM
from stephanie.models.plan_trace import ExecutionStepORM, PlanTraceORM
from stephanie.models.plan_trace_reuse_link import PlanTraceReuseLinkORM
from stephanie.models.plan_trace_revision import PlanTraceRevisionORM


class PlanTraceStore(BaseSQLAlchemyStore):
    orm_model = PlanTraceORM
    default_order_by = PlanTraceORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger=logger)
        self.name = "plan_traces"
        self.table_name = "plan_traces"

    def name(self) -> str:
        return self.name

    # --------------------
    # INSERT / UPDATE
    # --------------------
    def add(self, plan_trace: PlanTrace) -> int:
        def op():
            s = self._scope()
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
                self.logger.log("PlanTraceStored", {
                    "plan_trace_id": orm_trace.id,
                    "trace_id": orm_trace.trace_id,
                    "goal_id": orm_trace.goal_id,
                })
            return orm_trace.id
        return self._run(op)

    def update(self, plan_trace: PlanTrace) -> bool:
        def op():
            s = self._scope()
            orm_trace = s.query(PlanTraceORM).filter_by(trace_id=plan_trace.trace_id).first()
            if not orm_trace:
                if self.logger:
                    self.logger.log("PlanTraceUpdateFailed", {"error": "Trace not found", "trace_id": plan_trace.trace_id})
                return False
            orm_trace.final_output_text = plan_trace.final_output_text
            orm_trace.target_epistemic_quality = plan_trace.target_epistemic_quality
            orm_trace.target_epistemic_quality_source = plan_trace.target_epistemic_quality_source
            if isinstance(plan_trace.meta, dict):
                orm_trace.meta = {**(orm_trace.meta or {}), **plan_trace.meta}
            if plan_trace.execution_steps:
                orm_trace.execution_steps.clear()
                for i, step in enumerate(plan_trace.execution_steps, 1):
                    orm_trace.execution_steps.append(
                        ExecutionStepORM(
                            plan_trace=orm_trace,
                            pipeline_run_id=plan_trace.pipeline_run_id,
                            step_order=step.step_order or i,
                            step_id=str(step.step_id),
                            description=step.description,
                            output_text=step.output_text,
                            meta={**(step.attributes or {}), **(step.meta or {})},
                        )
                    )
            if self.logger:
                self.logger.log("PlanTraceUpdated", {
                    "trace_id": plan_trace.trace_id,
                    "step_count": len(plan_trace.execution_steps),
                })
            return True
        return self._run(op)

    # --------------------
    # RETRIEVAL
    # --------------------
    def get_by_id(self, trace_id: int) -> Optional[PlanTraceORM]:
        return self._run(lambda: self._scope().get(PlanTraceORM, trace_id))

    def get_by_trace_id(self, trace_id: str) -> Optional[PlanTraceORM]:
        return self._run(lambda: self._scope().query(PlanTraceORM).filter_by(trace_id=trace_id).first())

    def get_by_run_id(self, run_id: str) -> Optional[PlanTraceORM]:
        return self._run(lambda: self._scope().query(PlanTraceORM).filter_by(pipeline_run_id=run_id).first())

    def get_by_goal_id(self, goal_id: int) -> List[PlanTraceORM]:
        return self._run(lambda: self._scope().query(PlanTraceORM).filter_by(goal_id=goal_id).all())

    def get_traces_with_labels(self) -> List[PlanTraceORM]:
        return self._run(lambda: self._scope().query(PlanTraceORM).filter(PlanTraceORM.target_epistemic_quality.isnot(None)).all())

    def get_recent(self, limit: int = 10) -> List[PlanTraceORM]:
        return self._run(lambda: self._scope().query(PlanTraceORM).order_by(desc(PlanTraceORM.created_at)).limit(limit).all())

    def get_all(self, limit: int = 100) -> List[PlanTraceORM]:
        def op():
            q = self._scope().query(PlanTraceORM).order_by(desc(PlanTraceORM.created_at))
            return q.limit(limit).all() if limit else q.all()
        return self._run(op)

    def get_goal_text(self, trace_id: str) -> Optional[str]:
        def op():
            result = (
                self._scope().query(GoalORM.goal_text)
                .join(PlanTraceORM, GoalORM.id == PlanTraceORM.goal_id)
                .filter(PlanTraceORM.trace_id == trace_id)
                .first()
            )
            return result[0] if result else None
        return self._run(op)

    # --------------------
    # ANALYTICS / FILTERS
    # --------------------
    def get_similar_traces(self, query_text: str, top_k: int = 10, embedding=None) -> List[PlanTraceORM]:
        def op():
            if embedding is None:
                return []
            query_emb = embedding.get_or_create(query_text)
            traces = self.get_all(limit=500)
            scored = []
            for t in traces:
                candidate_text = (t.final_output_text or "") + " " + (t.plan_signature or "")
                cand_emb = embedding.get_or_create(candidate_text)
                sim = float(query_emb @ cand_emb / ((query_emb**2).sum()**0.5 * (cand_emb**2).sum()**0.5))
                scored.append((sim, t))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scored[:top_k]]
        return self._run(op)

    def get_by_goal_type(self, goal_type: str, limit: int = 50) -> List[PlanTraceORM]:
        def op():
            return (
                self._scope().query(PlanTraceORM)
                .join(GoalORM, GoalORM.id == PlanTraceORM.goal_id)
                .filter(GoalORM.goal_type == goal_type)
                .order_by(desc(PlanTraceORM.created_at))
                .limit(limit)
                .all()
            )
        return self._run(op)

    # --------------------
    # REUSE LINKS / REVISIONS
    # --------------------
    def add_reuse_link(self, parent_trace_id: str, child_trace_id: str):
        def op():
            if parent_trace_id == child_trace_id:
                if self.logger:
                    self.logger.log("PlanTraceReuseLinkSkipped", {"reason": "parent == child", "trace_id": parent_trace_id})
                return None
            link = PlanTraceReuseLinkORM(parent_trace_id=parent_trace_id, child_trace_id=child_trace_id)
            s = self._scope()
            s.add(link)
            s.flush()
            return link.id
        return self._run(op)

    def get_reuse_links_for_trace(self, trace_id: str):
        return self._run(lambda: self._scope().query(PlanTraceReuseLinkORM).filter(
            (PlanTraceReuseLinkORM.parent_trace_id == trace_id) |
            (PlanTraceReuseLinkORM.child_trace_id == trace_id)
        ).all())

    def add_revision(self, trace_id: str, revision_type: str, revision_text: str, source: str = "user"):
        def op():
            rev = PlanTraceRevisionORM(
                plan_trace_id=trace_id,
                revision_type=revision_type,
                revision_text=revision_text,
                source=source,
            )
            s = self._scope()
            s.add(rev)
            s.flush()
            return rev
        return self._run(op)

    def get_revisions(self, trace_id: str) -> list[PlanTraceRevisionORM]:
        return self._run(lambda: self._scope().query(PlanTraceRevisionORM).filter_by(plan_trace_id=trace_id).order_by(PlanTraceRevisionORM.created_at).all())

    # --------------------
    # EXECUTION STEPS
    # --------------------
    def get_step_by_id(self, step_id: int) -> Optional[ExecutionStepORM]:
        return self._run(lambda: self._scope().get(ExecutionStepORM, step_id))
