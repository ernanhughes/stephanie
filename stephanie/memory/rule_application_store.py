from typing import List, Optional

from sqlalchemy import and_, desc
from sqlalchemy.orm import Session

from stephanie.models.rule_application import RuleApplicationORM


class RuleApplicationStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "rule_applications"
        self.table_name = "rule_applications"

    def add(self, application: RuleApplicationORM) -> RuleApplicationORM:
        try:
            self.session.add(application)
            self.session.commit()
            self.session.refresh(application)
            if self.logger:
                self.logger.log("RuleApplicationAdded", {"rule_application_id": application.id})
            return application
        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("RuleApplicationAddFailed", {"error": str(e)})
            raise

    def get_by_id(self, application_id: int) -> Optional[RuleApplicationORM]:
        return self.session.query(RuleApplicationORM).get(application_id)

    def get_all(self) -> List[RuleApplicationORM]:
        return self.session.query(RuleApplicationORM).order_by(desc(RuleApplicationORM.created_at)).all()

    def get_by_goal(self, goal_id: int) -> List[RuleApplicationORM]:
        return (
            self.session.query(RuleApplicationORM)
            .filter(RuleApplicationORM.goal_id == goal_id)
            .order_by(desc(RuleApplicationORM.created_at))
            .all()
        )

    def get_by_hypothesis(self, hypothesis_id: int) -> List[RuleApplicationORM]:
        return (
            self.session.query(RuleApplicationORM)
            .filter(RuleApplicationORM.hypothesis_id == hypothesis_id)
            .order_by(desc(RuleApplicationORM.created_at))
            .all()
        )

    def get_by_pipeline_run(self, pipeline_run_id: int) -> List[RuleApplicationORM]:
        return (
            self.session.query(RuleApplicationORM)
            .filter(RuleApplicationORM.pipeline_run_id == pipeline_run_id)
            .order_by(desc(RuleApplicationORM.applied_at))
            .all()
        )

    def get_latest_for_run(self, pipeline_run_id: int) -> Optional[RuleApplicationORM]:
        return (
            self.session.query(RuleApplicationORM)
            .filter(RuleApplicationORM.pipeline_run_id == pipeline_run_id)
            .order_by(desc(RuleApplicationORM.applied_at))
            .first()
        )

    def get_for_goal_and_hypothesis(self, goal_id: int, hypothesis_id: int) -> List[RuleApplicationORM]:
        return (
            self.session.query(RuleApplicationORM)
            .filter(
                RuleApplicationORM.goal_id == goal_id,
                RuleApplicationORM.hypothesis_id == hypothesis_id,
            )
            .order_by(desc(RuleApplicationORM.applied_at))
            .all()
        )
