# stephanie/memory/rule_application_store.py
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import desc

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.rule_application import RuleApplicationORM


class RuleApplicationStore(BaseSQLAlchemyStore):
    orm_model = RuleApplicationORM
    default_order_by = RuleApplicationORM.applied_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "rule_applications"
        self.table_name = "rule_applications"

    def add(self, application: RuleApplicationORM) -> RuleApplicationORM:
        """Insert a new RuleApplication row."""
        def op(s):
            
            s.add(application)
            s.flush()
            if self.logger:
                self.logger.log(
                    "RuleApplicationAdded",
                    {"rule_application_id": application.id}
                )
            return application
        return self._run(op)

    def get_by_id(self, application_id: int) -> Optional[RuleApplicationORM]:
        def op(s):
            return self._scope().get(RuleApplicationORM, application_id)
        return self._run(op)

    def get_all(self) -> List[RuleApplicationORM]:
        def op(s):
            return (
                s.query(RuleApplicationORM)
                .order_by(desc(RuleApplicationORM.created_at))
                .all()
            )
        return self._run(op)

    def get_by_goal(self, goal_id: int) -> List[RuleApplicationORM]:
        def op(s):
            return (
                s.query(RuleApplicationORM)
                .filter(RuleApplicationORM.goal_id == goal_id)
                .order_by(desc(RuleApplicationORM.created_at))
                .all()
            )
        return self._run(op)

    def get_by_hypothesis(self, hypothesis_id: int) -> List[RuleApplicationORM]:
        def op(s):
            return (
                s.query(RuleApplicationORM)
                .filter(RuleApplicationORM.hypothesis_id == hypothesis_id)
                .order_by(desc(RuleApplicationORM.created_at))
                .all()
            )
        return self._run(op)

    def get_by_pipeline_run(self, pipeline_run_id: int) -> List[RuleApplicationORM]:
        def op(s):
            return (
                s.query(RuleApplicationORM)
                .filter(RuleApplicationORM.pipeline_run_id == pipeline_run_id)
                .order_by(desc(RuleApplicationORM.applied_at))
                .all()
            )
        return self._run(op)

    def get_latest_for_run(self, pipeline_run_id: int) -> Optional[RuleApplicationORM]:
        def op(s):
            return (
                s.query(RuleApplicationORM)
                .filter(RuleApplicationORM.pipeline_run_id == pipeline_run_id)
                .order_by(desc(RuleApplicationORM.applied_at))
                .first()
            )
        return self._run(op)

    def get_for_goal_and_hypothesis(
        self, goal_id: int, hypothesis_id: int
    ) -> List[RuleApplicationORM]:
        def op(s):
            return (
                s.query(RuleApplicationORM)
                .filter(
                    RuleApplicationORM.goal_id == goal_id,
                    RuleApplicationORM.hypothesis_id == hypothesis_id,
                )
                .order_by(desc(RuleApplicationORM.applied_at))
                .all()
            )
        return self._run(op)
