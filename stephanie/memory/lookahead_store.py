# stephanie/memory/lookahead_store.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.lookahead import LookaheadORM


class LookaheadStore(BaseSQLAlchemyStore):
    orm_model = LookaheadORM
    default_order_by = LookaheadORM.created_at.desc()
    
    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "lookahead"

    # -------------------
    # Insert
    # -------------------
    def insert(self, goal_id: int, result: LookaheadORM) -> int:
        """Insert a new lookahead result into the DB and return its ID."""
        def op(s):
            db_lookahead = LookaheadORM(
                goal_id=goal_id,
                agent_name=result.agent_name,
                model_name=result.model_name,
                input_pipeline=result.input_pipeline,
                suggested_pipeline=result.suggested_pipeline,
                rationale=result.rationale,
                reflection=result.reflection,
                backup_plans=json.dumps(result.backup_plans or []),
                extra_data=json.dumps(result.extra_data or {}),
                run_id=result.run_id,
                created_at=result.created_at or datetime.now(timezone.utc),
            )
            s.add(db_lookahead)
            s.flush()

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
        return self._run(op)

    # -------------------
    # Retrieval
    # -------------------
    def list_all(self, limit: int = 100) -> List[LookaheadORM]:
        """Get all stored lookaheads, newest first."""
        def op(s):
            return (
                s.query(LookaheadORM)
                .order_by(LookaheadORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return [self._orm_to_dataclass(r) for r in self._run(op)]

    def get_by_goal_id(self, goal_id: int) -> List[LookaheadORM]:
        def op(s):
            return (
                s.query(LookaheadORM)
                .filter_by(goal_id=goal_id)
                .order_by(LookaheadORM.created_at.desc())
                .all()
            )
        return [self._orm_to_dataclass(r) for r in self._run(op)]

    def get_by_run_id(self, run_id: str) -> Optional[LookaheadORM]:
        def op(s):
            return s.query(LookaheadORM).filter_by(run_id=run_id).first()
        row = self._run(op)
        return self._orm_to_dataclass(row) if row else None

    # -------------------
    # Conversion
    # -------------------
    def _orm_to_dataclass(self, row: LookaheadORM) -> LookaheadORM:
        return LookaheadORM(
            goal_id=row.goal_id,
            agent_name=row.agent_name,
            model_name=row.model_name,
            input_pipeline=row.input_pipeline,
            suggested_pipeline=row.suggested_pipeline,
            rationale=row.rationale,
            reflection=row.reflection,
            backup_plans=json.loads(row.backup_plans) if row.backup_plans else [],
            extra_data=json.loads(row.extra_data) if row.extra_data else {},
            run_id=row.run_id,
            created_at=row.created_at,
        )
