# stephanie/memory/symbolic_rule_store.py
from typing import List

import yaml
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from stephanie.models.evaluation import EvaluationORM
from stephanie.models.symbolic_rule import SymbolicRuleORM


class SymbolicRuleStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "symbolic_rules"
        self.table_name = "symbolic_rules"

    def insert(self, rule: SymbolicRuleORM):
        self.session.add(rule)
        self.session.commit()
        self.session.refresh(rule)
        return rule

    def get_all_rules(self) -> List[SymbolicRuleORM]:
        return self.session.query(SymbolicRuleORM).all()

    def get_rules_for_goal(self, goal) -> List[SymbolicRuleORM]:
        return (
            self.session.query(SymbolicRuleORM)
            .filter(SymbolicRuleORM.goal_id == goal.id)
            .all()
        )

    def get_applicable_rules(
        self, goal: dict, pipeline_run_id: int = None, config: dict = {}
    ):
        match_priority = config.get(
            "match_priority", ["goal_id", "pipeline_run_id", "metadata"]
        )
        metadata_mode = config.get("metadata_match_mode", "partial")
        allow_fallback = config.get("allow_fallback", True)

        filters = []

        if "goal_id" in match_priority and goal.get("id"):
            filters.append(SymbolicRuleORM.goal_id == goal["id"])

        if "pipeline_run_id" in match_priority and pipeline_run_id:
            filters.append(SymbolicRuleORM.pipeline_run_id == pipeline_run_id)

        if "metadata" in match_priority and allow_fallback:
            goal_type = goal.get("goal_type")
            goal_category = goal.get("goal_category")
            difficulty = goal.get("difficulty")

            if metadata_mode == "exact":
                filters.append(
                    and_(
                        SymbolicRuleORM.goal_type == goal_type,
                        SymbolicRuleORM.goal_category == goal_category,
                        SymbolicRuleORM.difficulty == difficulty,
                    )
                )
            elif metadata_mode == "partial":
                filters += [
                    and_(
                        SymbolicRuleORM.goal_type == goal_type,
                        SymbolicRuleORM.goal_category == None,
                        SymbolicRuleORM.difficulty == None,
                    ),
                    and_(
                        SymbolicRuleORM.goal_category == goal_category,
                        SymbolicRuleORM.goal_type == None,
                        SymbolicRuleORM.difficulty == None,
                    ),
                    and_(
                        SymbolicRuleORM.difficulty == difficulty,
                        SymbolicRuleORM.goal_type == None,
                        SymbolicRuleORM.goal_category == None,
                    ),
                ]

        if not filters:
            return []

        query = self.session.query(SymbolicRuleORM).filter(or_(*filters))
        return query.all()

    def find_matching_rules(self, goal) -> List[SymbolicRuleORM]:
        return (
            self.session.query(SymbolicRuleORM)
            .filter(
                SymbolicRuleORM.goal_id == goal.id
                # Add logic here if you want partial matches by goal_type, etc.
            )
            .order_by(SymbolicRuleORM.score.desc().nullslast())
            .all()
        )

    def update_rule_score(self, rule_id: int):
        scores = (
            self.session.query(EvaluationORM.score)
            .filter(EvaluationORM.symbolic_rule_id == rule_id)
            .all()
        )
        scores = [s[0] for s in scores if s[0] is not None]

        if scores:
            avg_score = sum(scores) / len(scores)
            rule = self.session.query(SymbolicRuleORM).get(rule_id)
            rule.score = avg_score
            self.session.commit()
            return avg_score
        return None

    def get_top_rules(self, top_k=10) -> List[SymbolicRuleORM]:
        return (
            self.session.query(SymbolicRuleORM)
            .order_by(SymbolicRuleORM.score.desc().nullslast())
            .limit(top_k)
            .all()
        )

    def track_pipeline_stage(self, stage_dict: dict, context: dict):
        # Only create if not already exists
        goal = context.get("goal")
        goal_id = goal.get("id")
        pipeline_run_id = context.get("pipeline_run_id")
        agent_name = stage_dict.get("name", "default_agent")
        rule_filter = {"agent_name": agent_name, "goal_id": goal_id}
        context_hash = SymbolicRuleORM.compute_context_hash(stage_dict, rule_filter)
        existing = (
            self.session.query(SymbolicRuleORM)
            .filter_by(
                goal_id=goal_id, context_hash=context_hash, agent_name=agent_name
            )
            .first()
        )

        if not existing:
            rule = SymbolicRuleORM(
                goal_id=goal_id,
                pipeline_run_id=pipeline_run_id,
                agent_name=agent_name,
                target="pipeline",
                attributes=stage_dict,
                filter=rule_filter,
                goal_type=goal.get("goal_type"),
                goal_category=goal.get("goal_category") or goal.get("focus_area"),
                difficulty=goal.get("difficulty"),
                context_hash=context_hash,
                source="pipeline_stage",
            )
            self.insert(rule)

    def load_from_yaml(self, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        for rule in data.get("rules", []):
            exists = (
                self.session.query(SymbolicRuleORM)
                .filter_by(
                    goal_type=rule.get("goal_type"),
                    agent_name=rule.get("agent_name"),
                    rule_text=rule.get("rule_text"),
                )
                .first()
            )
            if not exists:
                new_rule = SymbolicRuleORM(**rule)
                self.session.add(new_rule)
        self.session.commit()

    def get_all(self) -> list[SymbolicRuleORM]:
        try:
            rules = (
                self.session.query(SymbolicRuleORM)
                .order_by(SymbolicRuleORM.created_at.desc())
                .all()
            )
            if self.logger:
                self.logger.log("SymbolicRulesFetched", {"count": len(rules)})
            return rules
        except Exception as e:
            if self.logger:
                self.logger.log("SymbolicRulesFetchError", {"error": str(e)})
            return []

    def get_by_id(self, rule_id: int):
        try:
            return self.session.query(SymbolicRuleORM).filter_by(id=rule_id).first()
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "SymbolicRuleGetByIdError", {"rule_id": rule_id, "error": str(e)}
                )
            return None

    def exists_by_signature(self, signature: str) -> bool:
        """
        Checks if a symbolic rule exists by its signature hash.
        """
        return (
            self.session.query(SymbolicRuleORM)
            .filter_by(context_hash=signature)
            .first()
            is not None
        )

    def exists_similar(self, rule: SymbolicRuleORM, attr: str, new_val) -> bool:
        """
        Returns True if a rule already exists with the same target, attribute, and value.
        This prevents redundant mutations.
        """
        try:
            # Get all candidate rules with same agent, target, and optional goal
            query = self.session.query(SymbolicRuleORM).filter(
                SymbolicRuleORM.agent_name == rule.agent_name,
                SymbolicRuleORM.target == rule.target,
            )

            if rule.goal_id:
                query = query.filter(SymbolicRuleORM.goal_id == rule.goal_id)

            all_matches = query.all()

            # Check for exact match in attributes
            for existing_rule in all_matches:
                if (
                    existing_rule.attributes
                    and attr in existing_rule.attributes
                    and existing_rule.attributes[attr] == new_val
                ):
                    return True

            return False
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "SymbolicRuleExistsSimilarError",
                    {
                        "error": str(e),
                        "agent_name": rule.agent_name,
                        "attribute": attr,
                        "value": new_val,
                    },
                )
            return False
