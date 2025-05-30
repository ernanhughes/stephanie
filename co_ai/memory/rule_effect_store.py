from sqlalchemy.orm import Session
from datetime import datetime
from co_ai.models.rule_application import RuleApplicationORM

class RuleEffectStore:
    def __init__(self, db: Session, logger=None):
        self.db = db
        self.logger = logger

    def log_application(
        self,
        rule_id: int,
        goal_id: int,
        pipeline_run_id: int = None,
        hypothesis_id: int = None,
        result_score: float = None,
        result_label: str = None,
        applied_pipeline: list[str] = None,
        notes: str = None,
        extra_data: dict = None,
    ) -> RuleApplicationORM:
        try:
            application = RuleApplicationORM(
                rule_id=rule_id,
                goal_id=goal_id,
                pipeline_run_id=pipeline_run_id,
                hypothesis_id=hypothesis_id,
                result_score=result_score,
                result_label=result_label,
                applied_pipeline=applied_pipeline,
                notes=notes,
                extra_data=extra_data,
                created_at=datetime.utcnow(),
            )
            self.db.add(application)
            self.db.commit()
            self.db.refresh(application)

            if self.logger:
                self.logger.log("RuleApplicationLogged", application.to_dict())

            return application

        except Exception as e:
            self.db.rollback()
            if self.logger:
                self.logger.log("RuleApplicationError", {"error": str(e)})
            raise

    def get_by_rule(self, rule_id: int) -> list[RuleApplicationORM]:
        return self.db.query(RuleApplicationORM).filter_by(rule_id=rule_id).all()

    def get_recent(self, limit: int = 50) -> list[RuleApplicationORM]:
        return (
            self.db.query(RuleApplicationORM)
            .order_by(RuleApplicationORM.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_feedback_summary(self, rule_id: int) -> dict:
        results = (
            self.db.query(RuleApplicationORM.result_label)
            .filter(RuleApplicationORM.rule_id == rule_id)
            .all()
        )
        summary = {}
        for (label,) in results:
            if label:
                summary[label] = summary.get(label, 0) + 1
        return summary
This is her 