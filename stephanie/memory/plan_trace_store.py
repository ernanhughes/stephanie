# stephanie/memory/plan_trace_store.py
from __future__ import annotations

import traceback
from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

# Import the ORM
from stephanie.data.plan_trace import PlanTrace
from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.goal import GoalORM
from stephanie.models.plan_trace import ExecutionStepORM, PlanTraceORM
from stephanie.models.plan_trace_reuse_link import PlanTraceReuseLinkORM
from stephanie.models.plan_trace_revision import PlanTraceRevisionORM


class PlanTraceStore(BaseSQLAlchemyStore):
    """
    Store for managing PlanTraceORM objects in the database.
    Provides methods to insert, retrieve, update, and query plan traces.
    """
    orm_model = PlanTraceORM
    default_order_by = PlanTraceORM.created_at.desc()

    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "plan_traces"
        self.table_name = "plan_traces"

    def name(self) -> str:
        return self.name
    
    def add(self, plan_trace: PlanTrace) -> int:
        """
        Adds a new PlanTrace to the store.
        Converts the PlanTrace dataclass (and its ExecutionSteps) to ORM objects.
        """
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

        # Convert execution steps
        orm_steps = []
        for step in plan_trace.execution_steps:
            orm_step = ExecutionStepORM(
                plan_trace=orm_trace,   # sets FK automatically
                pipeline_run_id=plan_trace.pipeline_run_id,
                step_order=step.step_order or len(orm_steps) + 1,
                step_id=str(step.step_id),
                description=step.description,
                output_text=step.output_text,
                output_embedding_id=None,
                meta={**(step.attributes or {}), **(step.meta or {})},
            )
            orm_steps.append(orm_step)

        orm_trace.execution_steps = orm_steps

        return self.insert(orm_trace)
            
    def insert(self, plan_trace: PlanTraceORM) -> int:
        """
        Inserts a new PlanTraceORM into the database.
        Assumes the ORM object is already populated.
        Returns the ID of the inserted record.
        """
        try:
            self.session.add(plan_trace)
            self.session.flush()  # To get ID immediately before commit

            if self.logger:
                self.logger.log(
                    "PlanTraceStored",
                    {
                        "plan_trace_id": plan_trace.id,
                        "trace_id": plan_trace.trace_id,
                        "goal_id": plan_trace.goal_id,
                        "created_at": plan_trace.created_at.isoformat() if plan_trace.created_at else None,
                    },
                )
            
            self.session.commit()
            return plan_trace.id

        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceInsertFailed", {"error": str(e), "trace_id": getattr(plan_trace, 'trace_id', 'unknown')})
            raise

    def update(self, plan_trace: PlanTrace) -> bool:
        try:
            orm_trace = self.session.query(PlanTraceORM).filter(
                PlanTraceORM.trace_id == plan_trace.trace_id
            ).first()
            if not orm_trace:
                if self.logger:
                    self.logger.log("PlanTraceUpdateFailed", {
                        "error": "Trace not found",
                        "trace_id": plan_trace.trace_id
                    })
                return False

            # Update scalar fields
            orm_trace.final_output_text = plan_trace.final_output_text
            orm_trace.target_epistemic_quality = plan_trace.target_epistemic_quality
            orm_trace.target_epistemic_quality_source = plan_trace.target_epistemic_quality_source

            # Update scoring fields
            orm_trace.step_scores = (
                plan_trace.meta.get("step_scores")
                if isinstance(plan_trace.meta, dict)
                else None
            )

            # Update extra_data safely (merge, don’t overwrite)
            if not orm_trace.meta:
                orm_trace.meta = {}
            orm_trace.meta.update(plan_trace.meta or {})

            # Optionally update execution steps
            if plan_trace.execution_steps:
                # Clear and repopulate or merge
                orm_trace.execution_steps.clear()
                for step in plan_trace.execution_steps:
                    orm_trace.execution_steps.append(
                        ExecutionStepORM(
                            plan_trace=orm_trace,
                            pipeline_run_id=plan_trace.pipeline_run_id,
                            step_order=step.step_order,
                            step_id=str(step.step_id),
                            description=step.description,
                            output_text=step.output_text,
                            meta={**(step.attributes or {}), **(step.meta or {})},
                        )
                    )

            self.session.commit()
            self.logger.log("PlanTraceUpdated", {
                "trace_id": plan_trace.trace_id,
                    "step_count": len(plan_trace.execution_steps),
                })
            return True

        except Exception as e:
            self.session.rollback()
            self.logger.log("PlanTraceUpdateError", {
                "trace_id": plan_trace.trace_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return False

    def get_by_id(self, trace_id: int) -> Optional[PlanTraceORM]:
        """Retrieves a PlanTraceORM by its database ID."""
        try:
            return self.session.get(PlanTraceORM, trace_id)
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceGetByIdFailed", {"error": str(e), "id": trace_id})
            return None

    def get_by_trace_id(self, trace_id: str) -> Optional[PlanTraceORM]:
        """Retrieves a PlanTraceORM by its unique trace_id string."""
        try:
            return self.session.query(PlanTraceORM).filter(PlanTraceORM.trace_id == trace_id).first()
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceGetByTraceIdFailed", {"error": str(e), "trace_id": trace_id})
            return None


    def get_by_run_id(self, run_id: str) -> Optional[PlanTraceORM]:
        """Retrieves a PlanTraceORM by its unique run_id string."""
        try:
            print("Retrieving PlanTraceORM by run_id:", run_id)
            return self.session.query(PlanTraceORM).filter(PlanTraceORM.pipeline_run_id == run_id).first()
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceGetByRunIdFailed", {"error": str(e), "run_id": run_id})
            return None

    def get_by_goal_id(self, goal_id: int) -> List[PlanTraceORM]:
        """Retrieves all PlanTraceORMs associated with a specific goal."""
        try:
            return self.session.query(PlanTraceORM).filter(PlanTraceORM.goal_id == goal_id).all()
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceGetByGoalIdFailed", {"error": str(e), "goal_id": goal_id})
            return []

    def get_traces_with_labels(self) -> List[PlanTraceORM]:
        """Retrieves all PlanTraceORMs that have a target_epistemic_quality assigned (for training HRM)."""
        try:
            return self.session.query(PlanTraceORM).filter(PlanTraceORM.target_epistemic_quality.isnot(None)).all()
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceGetWithLabelsFailed", {"error": str(e)})
            return []

    def update_target_quality(self, trace_id: str, quality: float, source: str) -> bool:
        """Updates the target_epistemic_quality and its source for a given trace."""
        try:
            trace = self.get_by_trace_id(trace_id)
            if trace:
                trace.target_epistemic_quality = quality
                trace.target_epistemic_quality_source = source
                if self.logger:
                    self.logger.log("PlanTraceTargetQualityUpdated", {"trace_id": trace_id, "quality": quality, "source": source})
                return True
            else:
                if self.logger:
                    self.logger.log("PlanTraceTargetQualityUpdateFailed", {"reason": "Trace not found", "trace_id": trace_id})
                return False
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceTargetQualityUpdateError", {"error": str(e), "trace_id": trace_id})
            return False

    def get_recent(self, limit: int = 10) -> List[PlanTraceORM]:
        """Returns the most recently created PlanTraceORMs."""
        try:
            return self.session.query(PlanTraceORM).order_by(desc(PlanTraceORM.created_at)).limit(limit).all()
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceGetRecentFailed", {"error": str(e), "limit": limit})
            return []

    def get_all(self, limit: int = 100) -> List[PlanTraceORM]:
        """Returns PlanTraceORM records, limited by default to 100 unless overridden."""
        try:
            query = self.session.query(PlanTraceORM).order_by(desc(PlanTraceORM.created_at))
            if limit:
                query = query.limit(limit)
            return query.all()
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceGetAllFailed", {"error": str(e), "limit": limit})
            return []


    def get_goal_text(self, trace_id: str) -> Optional[str]:
        """
        Retrieve the goal text for a given plan_trace (by trace_id) using a direct join.
        More efficient than loading the full ORM relationship.
        """
        try:
            result = (
                self.session.query(GoalORM.goal_text)
                .join(PlanTraceORM, GoalORM.id == PlanTraceORM.goal_id)
                .filter(PlanTraceORM.trace_id == trace_id)
                .first()
            )
            if result:
                return result[0]  # since we're selecting only goal_text
            return None
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "PlanTraceGetGoalTextError",
                    {"error": str(e), "trace_id": trace_id},
                )
            return None



    def get_similar_traces(self, query_text: str, top_k: int = 10, embedding=None) -> List[PlanTraceORM]:
        """
        Retrieve PlanTraces most similar to the given query_text using embeddings.
        Requires `embedding` backend (defaults to memory.embedding).
        """
        try:
            if embedding is None:
                embedding = self.session.bind.memory.embedding  # fallback if memory is globally bound
            query_emb = embedding.get_or_create(query_text)

            traces = self.get_all(limit=500)  # pull candidates
            scored = []
            for t in traces:
                candidate_text = (t.final_output_text or "") + " " + (t.plan_signature or "")
                cand_emb = embedding.get_or_create(candidate_text)
                sim = float(query_emb @ cand_emb / ( (query_emb**2).sum()**0.5 * (cand_emb**2).sum()**0.5 ))
                scored.append((sim, t))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scored[:top_k]]

        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceSimilarityError", {"error": str(e), "query": query_text})
            return []

    def get_by_goal_type(self, goal_type: str, limit: int = 50) -> List[PlanTraceORM]:
        """Retrieve traces linked to a specific goal type (via GoalORM.goal_type)."""
        try:
            return (
                self.session.query(PlanTraceORM)
                .join(GoalORM, GoalORM.id == PlanTraceORM.goal_id)
                .filter(GoalORM.goal_type == goal_type)
                .order_by(desc(PlanTraceORM.created_at))
                .limit(limit)
                .all()
            )
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceGetByGoalTypeError", {"error": str(e), "goal_type": goal_type})
            return []

    def get_by_outcome(self, min_score: Optional[float] = None, success_only: bool = False, limit: int = 50) -> List[PlanTraceORM]:
        """
        Retrieve traces by outcome.
        - If min_score is set, filter by pipeline_score['overall'] >= min_score.
        - If success_only=True, filter traces with no error in meta.
        """
        try:
            query = self.session.query(PlanTraceORM)

            if min_score is not None:
                query = query.filter(
                    PlanTraceORM.pipeline_score["overall"].astext.cast(float) >= min_score
                )
            if success_only:
                query = query.filter(~PlanTraceORM.meta.has_key("error"))  # noqa: E711

            return query.order_by(desc(PlanTraceORM.created_at)).limit(limit).all()
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceGetByOutcomeError", {
                    "error": str(e),
                    "min_score": min_score,
                    "success_only": success_only
                })
            return []

    def add_reuse_link(self, parent_trace_id: str, child_trace_id: str):
        if parent_trace_id == child_trace_id:
            if self.logger:
                self.logger.log("PlanTraceReuseLinkSkipped", {
                    "reason": "parent == child",
                    "trace_id": parent_trace_id
                })
            return None
        link = PlanTraceReuseLinkORM(
            parent_trace_id=parent_trace_id,
            child_trace_id=child_trace_id
        )
        self.session.add(link)
        self.session.commit()
        if self.logger:
            self.logger.log("PlanTraceReuseLinkCreated", {
                "parent": parent_trace_id,
                "child": child_trace_id
            })
        return link.id

    def get_reuse_links_for_trace(self, trace_id: str):
        return (
            self.session.query(PlanTraceReuseLinkORM)
            .filter(
                (PlanTraceReuseLinkORM.parent_trace_id == trace_id) |
                (PlanTraceReuseLinkORM.child_trace_id == trace_id)
            )
            .all()
        )
    
    def add_revision(self, trace_id: str, revision_type: str, revision_text: str, source: str = "user"):
        revision = PlanTraceRevisionORM(
            plan_trace_id=trace_id,
            revision_type=revision_type,
            revision_text=revision_text,
            source=source,
        )
        self.session.add(revision)
        self.session.commit()
        return revision

    def get_revisions(self, trace_id: str) -> list[PlanTraceRevisionORM]:
        return (
            self.session.query(PlanTraceRevisionORM)
            .filter_by(plan_trace_id=trace_id)
            .order_by(PlanTraceRevisionORM.created_at)
            .all()
        )

    def get_step_by_id(self, step_id: int) -> Optional[ExecutionStepORM]:
        """Retrieve an ExecutionStepORM by its database ID."""
        try:
            return self.session.get(ExecutionStepORM, step_id)
        except Exception as e:
            if self.logger:
                self.logger.log("ExecutionStepGetByIdFailed", {"error": str(e), "id": step_id})
            return None