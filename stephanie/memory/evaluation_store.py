# stephanie/memory/evaluation_store.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import func

from stephanie.data.score_bundle import ScoreBundle
from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models import RuleApplicationORM
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.evaluation_rule_link import EvaluationRuleLinkORM
from stephanie.models.goal import GoalORM
from stephanie.models.score import ScoreORM
from stephanie.models.score_attribute import ScoreAttributeORM
from stephanie.scoring.scorable import Scorable


class EvaluationStore(BaseSQLAlchemyStore):
    orm_model = EvaluationORM
    default_order_by = "created_at"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "evaluations"
        self.table_name = "evaluations"

    def name(self) -> str:
        return self.name

    # -------------------
    # Insert
    # -------------------
    def insert(self, evaluation: EvaluationORM) -> int:
        """Insert a new Evaluation row and link to rule applications if relevant."""

        def op(s):
            s.add(evaluation)
            s.flush()

            # Link to rule applications if possible
            if evaluation.pipeline_run_id and evaluation.goal_id:
                rule_apps = (
                    s.query(RuleApplicationORM)
                    .filter_by(
                        pipeline_run_id=evaluation.pipeline_run_id,
                        goal_id=evaluation.goal_id,
                    )
                    .all()
                )
                for ra in rule_apps:
                    s.add(
                        EvaluationRuleLinkORM(
                            evaluation_id=evaluation.id,
                            rule_application_id=ra.id,
                        )
                    )
            return evaluation.id

        eval_id = self._run(op)

        if self.logger:
            self.logger.log(
                "ScoreStored",
                {
                    "evaluation_id": eval_id,
                    "goal_id": evaluation.goal_id,
                    "scorable_id": evaluation.scorable_id,
                    "scorable_type": evaluation.scorable_type,
                    "agent": evaluation.agent_name,
                    "model": evaluation.model_name,
                    "scores": evaluation.scores,
                    "timestamp": evaluation.created_at.isoformat()
                    if evaluation.created_at
                    else None,
                },
            )
        return eval_id

    def save_bundle(
        self,
        bundle: ScoreBundle,
        scorable: Scorable,
        context: dict,
        cfg: dict,
        source: str,
        embedding_type: str = "hnet",
        evaluator: Optional[str] = None,
        model_name=None,
        agent_name=None,
        *,
        container=None,  # needed for ScoreDeltaCalculator
    ) -> EvaluationORM:
        """Persist a full bundle (Evaluation + Scores + Attributes) into the DB,
        then handle delta logging and display.
        """

        def op(s):
            goal = context.get("goal")
            pipeline_run_id = context.get("pipeline_run_id")

            eval_orm = EvaluationORM(
                goal_id=goal.get("id") if goal else None,
                pipeline_run_id=pipeline_run_id,
                scorable_type=scorable.target_type,
                scorable_id=str(scorable.id),
                source=source,
                agent_name=agent_name or cfg.get("name"),
                model_name=model_name
                or cfg.get("model", {}).get("name", "UnknownModel"),
                embedding_type=embedding_type,
                evaluator_name=evaluator or "ScoreEvaluator",
                strategy=cfg.get("strategy"),
                reasoning_strategy=cfg.get("reasoning_strategy"),
                scores=bundle.to_dict(),  # keep snapshot in evaluation row
            )
            s.add(eval_orm)
            s.flush()

            # Save ScoreORM + attributes
            for result in bundle.results.values():
                score_orm = ScoreORM(
                    evaluation_id=eval_orm.id,
                    dimension=result.dimension,
                    score=result.score,
                    weight=result.weight,
                    source=result.source,
                    rationale=result.rationale,
                )
                s.add(score_orm)
                s.flush()

                # Attributes
                for k, v in (result.attributes or {}).items():
                    s.add(
                        ScoreAttributeORM(
                            score_id=score_orm.id,
                            key=k,
                            value=json.dumps(v)
                            if isinstance(v, (list, dict))
                            else str(v),
                            data_type=(
                                "json"
                                if isinstance(v, (list, dict))
                                else "float"
                                if isinstance(v, (int, float))
                                else "string"
                            ),
                        )
                    )

            return eval_orm

        eval_orm = self._run(op)

        # ---- Post-persistence hooks ----
        if self.logger:
            self.logger.log(
                "BundleSaved",
                {
                    "evaluation_id": eval_orm.id,
                    "goal_id": eval_orm.goal_id,
                    "scorable_id": eval_orm.scorable_id,
                    "scorable_type": eval_orm.scorable_type,
                    "bundle_keys": list(bundle.results.keys()),
                },
            )

        weighted_score = bundle.aggregate()

        # Log deltas if goal exists
        goal = context.get("goal")
        if goal and "id" in goal and container is not None:
            from stephanie.scoring.calculations.score_delta import \
                ScoreDeltaCalculator

            ScoreDeltaCalculator(
                cfg, self, container, self.logger
            ).log_score_delta(scorable, weighted_score, goal["id"])

        # Optional display
        from stephanie.scoring.score_display import ScoreDisplay

        ScoreDisplay.show(scorable, bundle.to_dict(), weighted_score)

        return eval_orm

    # -------------------
    # Retrieval
    # -------------------
    def get_by_goal_id(self, goal_id: int) -> list[dict]:
        return self._run(
            lambda s: [
                self._orm_to_dict(r)
                for r in s.query(EvaluationORM)
                .filter(EvaluationORM.goal_id == goal_id)
                .all()
            ]
        )

    def get_goal(self, eval_id: int) -> Optional[str]:
        def op(s):
            ev = s.query(EvaluationORM).filter_by(id=eval_id).first()
            if not ev:
                return None
            goal = s.query(GoalORM).filter_by(id=ev.goal_id).first()
            return goal.goal_text if goal else None

        return self._run(op)

    def get_by_goal_type(self, goal_type: str) -> list[dict]:
        return self._run(
            lambda s: [
                self._orm_to_dict(r)
                for r in s.query(EvaluationORM)
                .join(GoalORM)
                .filter(GoalORM.goal_type == goal_type)
                .all()
            ]
        )

    def get_by_run_id(self, run_id: str) -> list[dict]:
        return self._run(
            lambda s: [
                self._orm_to_dict(r)
                for r in s.query(EvaluationORM)
                .filter(EvaluationORM.pipeline_run_id == run_id)
                .all()
            ]
        )

    def get_by_pipeline_run_id(self, pipeline_run_id: int) -> list[dict]:
        return self.get_by_run_id(str(pipeline_run_id))

    def get_by_evaluator(self, evaluator_name: str) -> list[dict]:
        return self._run(
            lambda s: [
                self._orm_to_dict(r)
                for r in s.query(EvaluationORM)
                .filter(EvaluationORM.evaluator_name == evaluator_name)
                .all()
            ]
        )

    def get_by_strategy(self, strategy: str) -> list[dict]:
        return self._run(
            lambda s: [
                self._orm_to_dict(r)
                for r in s.query(EvaluationORM)
                .filter(EvaluationORM.strategy == strategy)
                .all()
            ]
        )

    def get_all(self, limit: int = 100) -> list[dict]:
        return self._run(
            lambda s: [
                self._orm_to_dict(r)
                for r in s.query(EvaluationORM)
                .order_by(EvaluationORM.created_at.desc())
                .limit(limit)
                .all()
            ]
        )

    def get_rules_for_score(self, evaluation_id: int) -> list[int]:
        return self._run(
            lambda s: [
                rid
                for (rid,) in s.query(
                    EvaluationRuleLinkORM.rule_application_id
                )
                .filter_by(evaluation_id=evaluation_id)
                .all()
            ]
        )

    def get_latest_score(self, scorable: Scorable, agent_name: str = None):
        def op(s):
            q = s.query(EvaluationORM).filter_by(
                scorable_type=scorable.target_type,
                scorable_id=str(scorable.id),
            )
            if agent_name:
                q = q.filter(EvaluationORM.agent_name == agent_name)
            latest = q.order_by(EvaluationORM.created_at.desc()).first()
            if latest and latest.scores:
                scores = (
                    latest.scores
                    if isinstance(latest.scores, dict)
                    else json.loads(latest.scores)
                )
                return scores.get("final_score")
            return None

        return self._run(op)

    def get_by_scorable(
        self,
        scorable_id: str | int,
        scorable_type: str,
        *,
        evaluator_name: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = 100,
        newest_first: bool = True,
    ) -> list[dict]:
        def op(s):
            q = s.query(EvaluationORM).filter(
                EvaluationORM.scorable_id == str(scorable_id),
                EvaluationORM.scorable_type == scorable_type,
            )
            if evaluator_name:
                q = q.filter(EvaluationORM.evaluator_name == evaluator_name)
            if since:
                q = q.filter(EvaluationORM.created_at >= since)
            q = q.order_by(
                EvaluationORM.created_at.desc()
                if newest_first
                else EvaluationORM.created_at.asc()
            )
            if limit is not None:
                q = q.limit(limit)
            return [self._orm_to_dict(r) for r in q.all()]

        return self._run(op)

    # -------------------
    # Helpers
    # -------------------
    def _orm_to_dict(self, row: EvaluationORM) -> dict:
        return {
            "id": row.id,
            "goal_id": row.goal_id,
            "scorable_id": row.scorable_id,
            "scorable_type": row.scorable_type,
            "agent_name": row.agent_name,
            "model_name": row.model_name,
            "evaluator_name": row.evaluator_name,
            "scores": (
                row.scores
                if isinstance(row.scores, dict)
                else json.loads(row.scores)
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
