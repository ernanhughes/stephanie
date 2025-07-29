# stephanie/memory/plan_trace_store.py

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

# Import the ORM
from stephanie.data.plan_trace import PlanTrace
from stephanie.models.plan_trace import PlanTraceORM
# Import related ORMs if needed for relationships/filters
# from stephanie.models.goal import GoalORM

class PlanTraceStore:
    """
    Store for managing PlanTraceORM objects in the database.
    Provides methods to insert, retrieve, and query plan traces.
    """
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "plan_traces"
        self.table_name = "plan_traces"

    def add(self, plan_trace: PlanTrace) -> int:
        """
        Adds a new PlanTrace to the store.
        Converts the PlanTrace dataclass to a PlanTraceORM and inserts it.
        Returns the ID of the inserted record.
        """
        orm_trace = PlanTraceORM(
            trace_id=plan_trace.trace_id,
            goal_id=plan_trace.goal_id,
            plan_signature=plan_trace.plan_signature,
            target_epistemic_quality=plan_trace.target_epistemic_quality,
            final_output_text=plan_trace.final_output_text,
            target_epistemic_quality_source=plan_trace.target_epistemic_quality_source,
            meta=plan_trace.extra_data,
        )
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
            
            # Commit is typically handled by the caller or session manager
            # to allow for batch inserts or rollback on error in a larger context.
            # self.session.commit() # Consider removing if caller handles transactions
            self.session.commit()  # Ensure the ORM object is flushed to get the I
            return plan_trace.id

        except Exception as e:
            # Let the caller handle rollback if needed, or do it here if this store manages transactions
            # self.session.rollback() 
            if self.logger:
                self.logger.log("PlanTraceInsertFailed", {"error": str(e), "trace_id": getattr(plan_trace, 'trace_id', 'unknown')})
            raise # Re-raise to let caller handle

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
                # Assuming session commit is handled externally or by a manager
                # self.session.commit() 
                if self.logger:
                    self.logger.log("PlanTraceTargetQualityUpdated", {"trace_id": trace_id, "quality": quality, "source": source})
                return True
            else:
                if self.logger:
                    self.logger.log("PlanTraceTargetQualityUpdateFailed", {"reason": "Trace not found", "trace_id": trace_id})
                return False
        except Exception as e:
            # self.session.rollback()
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

    def all(self) -> List[PlanTraceORM]:
        """Returns all PlanTraceORM records."""
        try:
            return self.session.query(PlanTraceORM).all()
        except Exception as e:
            if self.logger:
                self.logger.log("PlanTraceGetAllFailed", {"error": str(e)})
            return []
