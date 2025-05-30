# co_ai/symbolic/symbolic_rule_applier.py

from typing import Any, Dict, List
from co_ai.memory.symbolic_rule_store import SymbolicRuleORM
import yaml
from pathlib import Path

class SymbolicRuleApplier:
    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.enabled = cfg.get("symbolic", {}).get("enabled", False)
        self.rules = self._load_rules()

    def _load_rules(self):
        rules = []
        if self.cfg.symbolic.get("rules_file"):
            rules += self._load_rules_from_yaml(self.cfg.symbolic.rules_file)
        if self.cfg.symbolic.get("enable_db_rules", True):
            rules += self.memory.symbolic_rules.get_all_rules()
        return rules

    def _load_rules_from_yaml(self, path: str) -> list:
        if not Path(path).exists():
            self.logger.log("SymbolicRuleYAMLNotFound", {"path": path})
            return []

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        rules_list = raw.get("rules", raw)  # fallback to raw if it's already a list

        rules = []
        for item in rules_list:
            if isinstance(item, dict):
                rules.append(SymbolicRuleORM(**item))
            else:
                self.logger.log("InvalidSymbolicRuleFormat", {"item": str(item)})
        return rules

    def apply(self, context: dict) -> dict:
        if not self.enabled:
            return context

        goal = context.get("goal", {})
        pipeline_run_id = context.get("pipeline_run_id")
        current_pipeline = context.get("pipeline", [])

        matching_rules = [
            r for r in self.rules if self._matches_metadata(r, goal)
        ]

        if not matching_rules:
            self.logger.log("NoSymbolicRulesApplied", {"goal_id": goal.get("id")})
            return context

        self.logger.log("SymbolicRulesFound", {"count": len(matching_rules)})

        for rule in matching_rules:
            if rule.rule_text and "pipeline:" in rule.rule_text:
                suggested_pipeline = rule.rule_text.split("pipeline:")[-1].strip().split(",")
                suggested_pipeline = [s.strip() for s in suggested_pipeline if s.strip()]
                if suggested_pipeline:
                    self.logger.log("PipelineUpdatedBySymbolicRule", {
                        "from": current_pipeline,
                        "to": suggested_pipeline,
                        "rule_id": rule.id
                    })
                    context["pipeline"] = suggested_pipeline
                    context["pipeline_updated_by_symbolic_rule"] = True

            if rule.source == "lookahead" and rule.goal_type:
                context["symbolic_hint"] = f"use_{rule.goal_type.lower()}_strategy"

        return context

    def _matches_metadata(self, rule: SymbolicRuleORM, goal: Dict[str, Any]) -> bool:
        if rule.goal_id and rule.goal_id != goal.get("id"):
            return False
        if rule.goal_type and rule.goal_type != goal.get("goal_type"):
            return False
        if rule.goal_category and rule.goal_category != goal.get("goal_category"):
            return False
        if rule.difficulty and rule.difficulty != goal.get("difficulty"):
            return False
        if hasattr(goal, "focus_area") and rule.goal_category:
            if rule.goal_category != goal.get("focus_area"):  # fallback mapping
                return False
        return True
