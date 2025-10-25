# stephanie/memory/rule_effect_store.py
from __future__ import annotations

from typing import Dict, List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.rule_application import RuleApplicationORM


class RuleEffectStore(BaseSQLAlchemyStore):
    orm_model = RuleApplicationORM
    default_order_by = RuleApplicationORM.applied_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
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
        def op(s):
            
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
            s.add(application)
            s.flush()
            if self.logger:
                self.logger.log("RuleApplicationLogged", application.to_dict())
            return application
        return self._run(op)

    def get_by_rule(self, rule_id: int) -> List[RuleApplicationORM]:
        """Retrieve all applications for a given rule."""
        def op(s):
            return s.query(RuleApplicationORM).filter_by(rule_id=rule_id).all()
        return self._run(op)

    def get_recent(self, limit: int = 50) -> List[RuleApplicationORM]:
        """Get the most recent rule applications."""
        def op(s):
            return (
                s.query(RuleApplicationORM)
                .order_by(RuleApplicationORM.applied_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def get_feedback_summary(self, rule_id: int) -> Dict[str, int]:
        """Return a count of feedback labels for a specific rule."""
        def op(s):
            results = (
                s.query(RuleApplicationORM.result_label)
                .filter(RuleApplicationORM.rule_id == rule_id)
                .all()
            )
            summary: Dict[str, int] = {}
            for (label,) in results:
                if label:
                    summary[label] = summary.get(label, 0) + 1
            return summary
        return self._run(op)

    def get_by_run_and_goal(self, run_id: int, goal_id: int) -> List[RuleApplicationORM]:
        """Retrieve all rule applications for a specific pipeline run and goal."""
        if not run_id or not goal_id:
            if self.logger:
                self.logger.log(
                    "InvalidInputForRuleFetch",
                    {"reason": "Missing run_id or goal_id", "run_id": run_id, "goal_id": goal_id},
                )
            return []

        def op(s):
            apps = (
                s.query(RuleApplicationORM)
                .filter(
                    RuleApplicationORM.pipeline_run_id == int(run_id),
                    RuleApplicationORM.goal_id == int(goal_id),
                )
                .all()
            )
            if self.logger and apps:
                self.logger.log(
                    "RuleApplicationsFetched",
                    {"run_id": run_id, "goal_id": goal_id, "count": len(apps)},
                )
            return apps
        return self._run(op)

    def get_recent_performance(self, rule_id: int, limit: int = 10) -> List[Dict]:
        """Retrieve recent performance entries for a given rule."""
        def op(s):
            entries = (
                s.query(RuleApplicationORM)
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
        return self._run(op)
