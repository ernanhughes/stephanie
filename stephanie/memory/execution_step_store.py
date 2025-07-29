# stephanie/memory/execution_step_store.py

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import asc

# Import the ORM
from stephanie.data.plan_trace import ExecutionStep
from stephanie.models.plan_trace import ExecutionStepORM

class ExecutionStepStore:
    """
    Store for managing ExecutionStepORM objects in the database.
    Provides methods to insert, retrieve, and query execution steps.
    """
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "execution_steps"
        self.table_name = "execution_steps"

    def add(self, step: ExecutionStep) -> int:
        """
        Adds a new ExecutionStep to the store.
        Converts the ExecutionStep dataclass to an ExecutionStepORM and inserts it.
        Returns the ID of the inserted record.
        """
        orm_step = ExecutionStepORM(
            step_id=step.step_id,
            plan_trace_id=step.plan_trace_id,
            step_order=step.step_order,  
            description=step.description,
            output_text=step.output_text,
            meta=step.extra_data,
        )
        return self.insert(orm_step)

    def insert(self, step: ExecutionStepORM) -> int:
        """
        Inserts a new ExecutionStepORM into the database.
        Assumes the ORM object is already populated.
        Returns the ID of the inserted record.
        """
        try:
            self.session.add(step)
            self.session.flush() # To get ID immediately

            if self.logger:
                self.logger.log(
                    "ExecutionStepStored",
                    {
                        "step_id": step.id,
                        "plan_trace_id": step.plan_trace_id,
                        "step_order": step.step_order,
                        "step_id_str": step.step_id, # Log the application-level ID
                        "created_at": step.created_at.isoformat() if step.created_at else None,
                    },
                )
            
            # self.session.commit() # Consider transaction management
            return step.id

        except Exception as e:
            # self.session.rollback()
            if self.logger:
                self.logger.log("ExecutionStepInsertFailed", {"error": str(e), "step_id": getattr(step, 'step_id', 'unknown')})
            raise

    def get_by_id(self, step_id: int) -> Optional[ExecutionStepORM]:
        """Retrieves an ExecutionStepORM by its database ID."""
        try:
            return self.session.get(ExecutionStepORM, step_id)
        except Exception as e:
            if self.logger:
                self.logger.log("ExecutionStepGetByIdFailed", {"error": str(e), "id": step_id})
            return None

    def get_steps_by_trace_id(self, plan_trace_id: int, ordered: bool = True) -> List[ExecutionStepORM]:
        """Retrieves all ExecutionStepORMs for a given PlanTraceORM ID."""
        try:
            query = self.session.query(ExecutionStepORM).filter(ExecutionStepORM.plan_trace_id == plan_trace_id)
            if ordered:
                query = query.order_by(asc(ExecutionStepORM.step_order))
            return query.all()
        except Exception as e:
            if self.logger:
                self.logger.log("ExecutionStepsGetByTraceIdFailed", {"error": str(e), "plan_trace_id": plan_trace_id})
            return []

    def get_by_evaluation_id(self, evaluation_id: int) -> Optional[ExecutionStepORM]:
        """Retrieves an ExecutionStepORM by its linked EvaluationORM ID."""
        try:
            return self.session.query(ExecutionStepORM).filter(ExecutionStepORM.evaluation_id == evaluation_id).first()
        except Exception as e:
            if self.logger:
                self.logger.log("ExecutionStepGetByEvaluationIdFailed", {"error": str(e), "evaluation_id": evaluation_id})
            return None

    # Add more query methods as needed, e.g., steps by step_id string, by date, etc.
    def get_all(self) -> List[ExecutionStepORM]:
        """Retrieves all ExecutionStepORMs in the database."""
        try:
            return self.session.query(ExecutionStepORM).all()
        except Exception as e:
            if self.logger:
                self.logger.log("ExecutionStepsGetAllFailed", {"error": str(e)})
            return []