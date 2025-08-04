# stephanie/memory/evaluation_store.py
# stores/score_store.py
import json
from typing import Optional

from sqlalchemy.orm import Session

from stephanie.models import RuleApplicationORM
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.evaluation_rule_link import EvaluationRuleLinkORM
from stephanie.models.goal import GoalORM
from stephanie.scoring.scorable import Scorable


class EvaluationStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "evaluations"
        self.table_name = "evaluations"

    def insert(self, evaluation: EvaluationORM):
        """
        Inserts a new score into the database.
        Accepts a dictionary (e.g., from Score dataclass).
        """
        try:
            self.session.add(evaluation)
            self.session.flush()  # To get ID immediately

            if self.logger:
                self.logger.log(
                    "ScoreStored",
                    {
                        "evaluation_id": evaluation.id,
                        "goal_id": evaluation.goal_id,
                        "hypothesis_id": evaluation.hypothesis_id,
                        "agent": evaluation.agent_name,
                        "model": evaluation.model_name,
                        "scores": evaluation.scores,
                        "timestamp": evaluation.created_at.isoformat(),
                    },
                )

            # Link score to rule application if possible
            if evaluation.pipeline_run_id and evaluation.goal_id:
                rule_apps = (
                    self.session.query(RuleApplicationORM)
                    .filter_by(
                        pipeline_run_id=evaluation.pipeline_run_id,
                        goal_id=evaluation.goal_id,
                    )
                    .all()
                )
                for ra in rule_apps:
                    link = EvaluationRuleLinkORM(
                        evaluation_id=evaluation.id, rule_application_id=ra.id
                    )
                    self.session.add(link)
                self.logger.log(
                    "ScoreLinkedToRuleApplications",
                    {
                        "score_id": evaluation.id,
                        "linked_rule_application_ids": [ra.id for ra in rule_apps],
                    },
                )

            self.session.refresh(evaluation)
            self.session.commit()
            return evaluation.id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("ScoreInsertFailed", {"error": str(e)})
            raise

    def get_by_goal_id(self, goal_id: int) -> list[dict]:
        """Returns all scores associated with a specific goal."""
        results = (
            self.session.query(EvaluationORM)
            .join(GoalORM)
            .filter(GoalORM.id == goal_id)
            .all()
        )
        return [self._orm_to_dict(r) for r in results]

    def get_goal(self, id: int) -> Optional[str]:
        """Returns the goal text for a given goal ID."""
        evaluation = self.session.query(EvaluationORM).filter(EvaluationORM.id == id).first()
        goal = self.session.query(GoalORM).filter(GoalORM.id == evaluation.goal_id).first()
        return goal if goal else None

    def get_by_goal_type(self, goal_type: str) -> list[dict]:
        """Returns all scores associated with a specific goal."""
        results = (
            self.session.query(EvaluationORM)
            .join(GoalORM)
            .filter(GoalORM.goal_type == goal_type)
            .all()
        )
        return [self._orm_to_dict(r) for r in results]

    def get_by_hypothesis_id(
        self, hypothesis_id: int, source: Optional[str] = None
    ) -> list[dict]:
        """Returns all scores associated with a specific hypothesis, optionally filtered by evaluator source."""
        query = self.session.query(EvaluationORM).filter(
            EvaluationORM.hypothesis_id == hypothesis_id
        )

        if source:
            query = query.filter(EvaluationORM.evaluator_name == source)

        results = query.all()
        return [self._orm_to_dict(r) for r in results]

    def get_by_run_id(self, run_id: str) -> list[dict]:
        """Returns all scores associated with a specific pipeline run."""
        results = (
            self.session.query(EvaluationORM)
            .filter(EvaluationORM.run_id == run_id)
            .all()
        )
        return [self._orm_to_dict(r) for r in results]

    def get_by_pipeline_run_id(self, pipeline_run_id: int) -> list[dict]:
        """Returns all scores associated with a specific pipeline run."""
        results = (
            self.session.query(EvaluationORM)
            .filter(EvaluationORM.pipeline_run_id == pipeline_run_id)
            .all()
        )
        return [self._orm_to_dict(r) for r in results]

    def get_by_evaluator(self, evaluator_name: str) -> list[dict]:
        """Returns all scores produced by a specific evaluator (LLM, MRQ, etc.)"""
        results = (
            self.session.query(EvaluationORM)
            .filter(EvaluationORM.evaluator_name == evaluator_name)
            .all()
        )
        return [self._orm_to_dict(r) for r in results]

    def get_by_strategy(self, strategy: str) -> list[dict]:
        """Returns all scores generated using a specific reasoning strategy."""
        results = (
            self.session.query(EvaluationORM)
            .filter(EvaluationORM.strategy == strategy)
            .all()
        )
        return [self._orm_to_dict(r) for r in results]

    def get_all(self, limit: int = 100) -> list[dict]:
        """Returns the most recent scores up to a limit."""
        results = (
            self.session.query(EvaluationORM)
            .order_by(EvaluationORM.created_at.desc())
            .limit(limit)
            .all()
        )
        return [self._orm_to_dict(r) for r in results]

    def _orm_to_dict(self, row: EvaluationORM) -> dict:
        """Converts an ORM object back to a dictionary format"""
        return {
            "id": row.id,
            "goal_id": row.goal_id,
            "hypothesis_id": row.hypothesis_id,
            "agent_name": row.agent_name,
            "model_name": row.model_name,
            "evaluator_name": row.evaluator_name,
            "scores": (
                row.scores if isinstance(row.scores, dict) else json.loads(row.scores)
            )
            if row.scores
            else {},
            "strategy": row.strategy,
            "reasoning_strategy": row.reasoning_strategy,
            "pipeline_run_id": row.pipeline_run_id,
            "extra_data": (
                row.extra_data
                if isinstance(row.extra_data, dict)
                else json.loads(row.extra_data)
            )
            if row.extra_data
            else {},
            "created_at": row.created_at,
        }

    def get_rules_for_score(self, score_id: int) -> list[int]:
        links = (
            self.session.query(EvaluationRuleLinkORM.rule_application_id)
            .filter_by(score_id=score_id)
            .all()
        )
        return [rid for (rid,) in links]

    def get_by_hypothesis_ids(self, hypothesis_ids: list[int]) -> list[EvaluationORM]:
        if not hypothesis_ids:
            return []
        try:
            return (
                self.session.query(EvaluationORM)
                .filter(EvaluationORM.hypothesis_id.in_(hypothesis_ids))
                .all()
            )
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "EvaluationStoreError",
                    {
                        "method": "get_by_hypothesis_ids",
                        "error": str(e),
                        "hypothesis_ids": hypothesis_ids,
                    },
                )
            return []

    def get_latest_score(self, scorable: Scorable, agent_name: str = None):
        query = self.session.query(EvaluationORM).filter_by(
            target_type=scorable.target_type, target_id=str(scorable.id)
        )

        if agent_name:
            query = query.filter(EvaluationORM.agent_name == agent_name)

        query = query.order_by(EvaluationORM.created_at.desc())
        latest = query.first()

        if latest and latest.scores:
            return latest.scores.get("final_score")

        return None
