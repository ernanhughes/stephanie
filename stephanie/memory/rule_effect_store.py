# stephanie/memory/rule_effect_store.py
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from stephanie.models.rule_application import RuleApplicationORM


class RuleEffectStore:
    def __init__(self, db: Session, logger=None):
        self.db = db
        self.logger = logger
        self.name = "rule_effects"
        self.table_name = "rule_applications"

    def insert(
        self,
        rule_id: int,
        goal_id: int,
        pipeline_run_id: Optional[int] = None,
        hypothesis_id: Optional[int] = None,
        result_score: Optional[float] = None,
        change_type: Optional[str] = None,
        agent_name: Optional[str] = None,
        notes: Optional[str] = None,
        details: Optional[Dict] = None,
        stage_details: Optional[Dict] = None,
        context_hash: Optional[str] = None,
    ) -> RuleApplicationORM:
        """Insert a new rule application record into the database."""
        try:
            application = RuleApplicationORM(
                rule_id=rule_id,
                goal_id=goal_id,
                pipeline_run_id=pipeline_run_id,
                hypothesis_id=hypothesis_id,
                post_score=result_score,
                change_type=change_type,
                agent_name=agent_name,
                notes=notes,
                details=details,
                stage_details=stage_details,
                context_hash=context_hash,
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

    def get_by_rule(self, rule_id: int) -> List[RuleApplicationORM]:
        """Retrieve all applications for a given rule."""
        return self.db.query(RuleApplicationORM).filter_by(rule_id=rule_id).all()

    def get_recent(self, limit: int = 50) -> List[RuleApplicationORM]:
        """Get the most recent rule applications."""
        return (
            self.db.query(RuleApplicationORM)
            .order_by(RuleApplicationORM.applied_at.desc())
            .limit(limit)
            .all()
        )

    def get_feedback_summary(self, rule_id: int) -> Dict[str, int]:
        """Return a count of feedback labels for a specific rule."""
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

    def get_by_run_and_goal(
        self, run_id: int, goal_id: int
    ) -> List[RuleApplicationORM]:
        """
        Retrieve all rule applications for a specific pipeline run and goal.

        Args:
            run_id (int): The ID of the pipeline run.
            goal_id (int): The ID of the goal.

        Returns:
            List[RuleApplicationORM]: Matching rule applications.
        """
        if not run_id or not goal_id:
            if self.logger:
                self.logger.log(
                    "InvalidInputForRuleFetch",
                    {
                        "reason": "Missing run_id or goal_id",
                        "run_id": run_id,
                        "goal_id": goal_id,
                    },
                )
            return []

        try:
            applications = (
                self.db.query(RuleApplicationORM)
                .filter(
                    RuleApplicationORM.pipeline_run_id == int(run_id),
                    RuleApplicationORM.goal_id == int(goal_id),
                )
                .all()
            )

            if self.logger and len(applications) > 0:
                self.logger.log(
                    "RuleApplicationsFetched",
                    {"run_id": run_id, "goal_id": goal_id, "count": len(applications)},
                )

            return applications

        except Exception as e:
            if self.logger:
                self.logger.log(
                    "RuleApplicationFetchError",
                    {"error": str(e), "run_id": run_id, "goal_id": goal_id},
                )
            return []

    def get_recent_performance(self, rule_id: int, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent performance entries for a given rule.

        Args:
            rule_id (int): The ID of the rule.
            limit (int): How many recent entries to return.

        Returns:
            List[Dict]: Recent performance data with score, timestamp, and optional metadata.
        """
        try:
            entries = (
                self.db.query(RuleApplicationORM)
                .filter(RuleApplicationORM.rule_id == rule_id)
                .order_by(RuleApplicationORM.applied_at.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "score": e.post_score,
                    "applied_at": e.applied_at.isoformat() if e.applied_at else None,
                    "agent": e.agent_name,
                    "change_type": e.change_type,
                    "context_hash": e.context_hash,
                    "details": e.details,
                }
                for e in entries
            ]

        except Exception as e:
            if self.logger:
                self.logger.log(
                    "RecentPerformanceError", {"error": str(e), "rule_id": rule_id}
                )
            return []
