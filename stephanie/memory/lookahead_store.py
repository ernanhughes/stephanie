# stores/lookahead_store.py
import json
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from stephanie.models.lookahead import LookaheadORM


class LookaheadStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "lookahead"
    
    def name(self) -> str:
        return "lookahead"

    def insert(self, goal_id: int, result: LookaheadORM):
        """
        Inserts a new lookahead result into the database.
        Assumes goal already exists.
        """
        try:
            # Build ORM object
            db_lookahead = LookaheadORM(
                goal_id=goal_id,
                agent_name=result.agent_name,
                model_name=result.model_name,
                input_pipeline=result.input_pipeline,
                suggested_pipeline=result.suggested_pipeline,
                rationale=result.rationale,
                reflection=result.reflection,
                backup_plans=json.dumps(result.backup_plans) if result.backup_plans else None,
                extra_data=json.dumps(result.extra_data or {}),
                run_id=result.run_id,
                created_at=result.created_at or datetime.now(timezone.utc),
            )

            self.session.add(db_lookahead)
            self.session.flush()  # To get ID immediately

            if self.logger:
                self.logger.log(
                    "LookaheadInserted",
                    {
                        "goal_id": goal_id,
                        "agent": result.agent_name,
                        "model": result.model_name,
                        "pipeline": result.input_pipeline,
                        "suggested_pipeline": result.suggested_pipeline,
                        "rationale_snippet": (result.rationale or "")[:100],
                    },
                )

            return db_lookahead.id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("LookaheadInsertFailed", {"error": str(e)})
            raise

    def list_all(self, limit: int = 100) -> List[LookaheadORM]:
        """Returns all stored lookaheads, converted back to dataclass"""
        db_results = self.session.query(LookaheadORM).order_by(LookaheadORM.created_at.desc()).limit(limit).all()
        return [self._orm_to_dataclass(result) for result in db_results]

    def get_by_goal_id(self, goal_id: int) -> List[LookaheadORM]:
        results = (
            self.session.query(LookaheadORM)
            .filter_by(goal_id=goal_id)
            .order_by(LookaheadORM.created_at.desc())
            .all()
        )
        return [self._orm_to_dataclass(r) for r in results]

    def get_by_run_id(self, run_id: str) -> Optional[LookaheadORM]:
        result = self.session.query(LookaheadORM).filter_by(run_id=run_id).first()
        return self._orm_to_dataclass(result) if result else None

    def _orm_to_dataclass(self, row: LookaheadORM) -> LookaheadORM:
        return LookaheadORM(
            goal=row.goal_id,
            agent_name=row.agent_name,
            model_name=row.model_name,
            input_pipeline=row.input_pipeline,
            suggested_pipeline=row.suggested_pipeline,
            rationale=row.rationale,
            reflection=row.reflection,
            backup_plans=json.loads(row.backup_plans) if row.backup_plans else [],
            metadata=json.loads(row.extra_data) if row.extra_data else {},
            run_id=row.run_id,
            created_at=row.created_at,
        )