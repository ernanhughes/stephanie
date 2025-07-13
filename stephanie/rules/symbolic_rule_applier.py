# stephanie/rules/symbolic_rule_applier.py
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from stephanie.memory.symbolic_rule_store import SymbolicRuleORM


class SymbolicRuleApplier:
    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.enabled = cfg.get("symbolic", {}).get("enabled", False)
        self._rules = self._load_rules() if self.enabled else []

    @property
    def rules(self) -> list:
        return self._rules

    def apply(self, context: dict) -> dict:
        if not self.enabled:
            return context

        goal = context.get("goal", {})
        pipeline_run_id = context.get("pipeline_run_id")
        current_pipeline = context.get("pipeline", [])

        matching_rules = [r for r in self.rules if self._matches_metadata(r, goal)]

        if not matching_rules:
            self.logger.log("NoSymbolicRulesApplied", {"goal_id": goal.get("id")})
            return context

        self.logger.log("SymbolicRulesFound", {"count": len(matching_rules)})

        for rule in matching_rules:
            if rule.rule_text and "pipeline:" in rule.rule_text:
                suggested_pipeline = (
                    rule.rule_text.split("pipeline:")[-1].strip().split(",")
                )
                suggested_pipeline = [
                    s.strip() for s in suggested_pipeline if s.strip()
                ]
                if suggested_pipeline:
                    self.logger.log(
                        "PipelineUpdatedBySymbolicRule",
                        {
                            "from": current_pipeline,
                            "to": suggested_pipeline,
                            "rule_id": rule.id,
                        },
                    )
                    context["pipeline"] = suggested_pipeline
                    context["pipeline_updated_by_symbolic_rule"] = True

            if rule.source == "lookahead" and rule.goal_type:
                context["symbolic_hint"] = f"use_{rule.goal_type.lower()}_strategy"

        return context

    from datetime import datetime

    from stephanie.models import RuleApplicationORM

    def apply_to_agent(self, cfg: Dict, context: Dict) -> Dict:
        if not self.enabled:
            return cfg

        goal = context.get("goal", {})
        pipeline_run_id = context.get("pipeline_run_id")
        agent_name = cfg.get("name")

        matching_rules = [
            r
            for r in self.rules
            if r.agent_name == agent_name and self._matches_metadata(r, goal)
        ]

        if not matching_rules:
            self.logger.log(
                "NoSymbolicAgentRulesApplied",
                {
                    "agent": agent_name,
                    "goal_id": goal.get("id"),
                },
            )
            return cfg

        self.logger.log(
            "SymbolicAgentRulesFound",
            {
                "agent": agent_name,
                "goal_id": goal.get("id"),
                "count": len(matching_rules),
            },
        )

        for rule in matching_rules:
            # Apply new-style attributes
            if rule.attributes:
                for key, value in rule.attributes.items():
                    if key in cfg:
                        self.logger.log(
                            "SymbolicAgentOverride",
                            {
                                "agent": agent_name,
                                "key": key,
                                "old_value": cfg[key],
                                "new_value": value,
                                "rule_id": rule.id,
                            },
                        )
                    else:
                        self.logger.log(
                            "SymbolicAgentNewKey",
                            {
                                "agent": agent_name,
                                "key": key,
                                "value": value,
                                "rule_id": rule.id,
                            },
                        )
                    cfg[key] = value

            # Apply legacy rule_text (optional, for backward compatibility)
            if rule.rule_text:
                entries = [e.strip() for e in rule.rule_text.split(",") if e.strip()]
                for entry in entries:
                    if ":" in entry:
                        key, value = [s.strip() for s in entry.split(":", 1)]
                        if key in cfg:
                            self.logger.log(
                                "SymbolicAgentOverride",
                                {
                                    "agent": agent_name,
                                    "key": key,
                                    "old_value": cfg[key],
                                    "new_value": value,
                                    "rule_id": rule.id,
                                },
                            )
                        else:
                            self.logger.log(
                                "SymbolicAgentNewKey",
                                {
                                    "agent": agent_name,
                                    "key": key,
                                    "value": value,
                                    "rule_id": rule.id,
                                },
                            )
                        cfg[key] = value

            # Record the application of this rule
            self.memory.rule_effects.insert(
                goal_id=goal.get("id"),
                agent_name=agent_name,
                rule_id=rule.id,
                pipeline_run_id=pipeline_run_id,
                details=rule.to_dict(),
                stage_details=cfg,
            )

        return cfg

    def apply_prompt_rules(
        self, agent_name: str, prompt_cfg: dict, context: dict
    ) -> dict:
        """
        Applies prompt-level symbolic rules to the prompt config before generation.

        Returns the updated prompt_cfg.
        """
        goal = context.get("goal", {})
        applicable_rules = [
            rule
            for rule in self.rules
            if rule.agent_name == agent_name
            # and self._matches_filter(rule.filter, goal)
        ]

        if not applicable_rules:
            self.logger.log("NoPromptRulesFound", {"agent": agent_name})
            return prompt_cfg

        for rule in applicable_rules:
            for key, value in rule.attributes.items():
                self.logger.log(
                    "PromptAttributeOverride",
                    {
                        "agent": agent_name,
                        "key": key,
                        "old_value": prompt_cfg.get(key),
                        "new_value": value,
                        "rule_id": rule.id,
                        "emoji": "🛠️",
                    },
                )
                self.set_nested(prompt_cfg, key, value)

            # Optional: record the rule application
            self.memory.rule_effects.insert(
                rule_id=rule.id,
                goal_id=goal.get("id"),
                pipeline_run_id=context.get("pipeline_run_id"),
                details=prompt_cfg,
            )

        return prompt_cfg

    def set_nested(self, cfg: dict, dotted_key: str, value):
        keys = dotted_key.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def apply_to_prompt(self, cfg: Dict, context: Dict) -> Dict:
        if not self.enabled:
            return cfg

        goal = context.get("goal", {})
        pipeline_run_id = context.get("pipeline_run_id")
        prompt_name = cfg.get("prompt_key", "unknown_prompt")

        matching_rules = [
            r
            for r in self.rules
            if r.target == "prompt" and self._matches_filter(r.filter, goal)
        ]

        if not matching_rules:
            self.logger.log(
                "NoSymbolicPromptRulesApplied",
                {
                    "prompt": prompt_name,
                    "goal_id": goal.get("id"),
                },
            )
            return cfg

        self.logger.log(
            "SymbolicPromptRulesFound",
            {
                "prompt": prompt_name,
                "goal_id": goal.get("id"),
                "count": len(matching_rules),
            },
        )

        for rule in matching_rules:
            for key, value in rule.attributes.items():
                if key in cfg:
                    self.logger.log(
                        "SymbolicPromptOverride",
                        {
                            "prompt": prompt_name,
                            "key": key,
                            "old_value": cfg[key],
                            "new_value": value,
                            "rule_id": rule.id,
                        },
                    )
                else:
                    self.logger.log(
                        "SymbolicPromptNewKey",
                        {
                            "prompt": prompt_name,
                            "key": key,
                            "value": value,
                            "rule_id": rule.id,
                        },
                    )
                cfg[key] = value

            # Track the application of the prompt-level rule
            self.memory.rule_effects.insert(
                rule_id=rule.id,
                goal_id=goal.get("id"),
                pipeline_run_id=pipeline_run_id,
                agent_name=cfg.get("name", "prompt"),
                context_hash=self.compute_context_hash(context),
                run_id=context.get("run_id"),
            )

        return cfg

    def _matches_filter(self, filter_dict: dict, target_obj: dict) -> bool:
        """Generic matcher for symbolic rule filters"""
        for key, value in filter_dict.items():
            target_value = target_obj.get(key)
            if isinstance(value, list):
                if target_value not in value:
                    return False
            else:
                if target_value != value:
                    return False
        return True

    def track_pipeline_stage(self, stage_dict: dict, context: dict):
        self.memory.symbolic_rules.track_pipeline_stage(stage_dict, context)

    def get_nested_value(d, key_path: str):
        keys = key_path.split(".")
        for key in keys:
            d = d.get(key, {})
        return d if d else None

    def set_nested_value(d, key_path: str, value):
        keys = key_path.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _load_rules(self):
        rules = []
        symbolic_dict = self.cfg.get("symbolic", {})
        if symbolic_dict.get("rules_file"):
            rules += self._load_rules_from_yaml(symbolic_dict.get("rules_file"))
        if symbolic_dict.get("enable_db_rules", True):
            rules += self.memory.symbolic_rules.get_all_rules()
        return rules

    def _load_rules_from_yaml(self, path: str) -> list:
        if not Path(path).exists():
            self.logger.log("SymbolicRuleYAMLNotFound", {"path": path})
            return []

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        rules_list = raw.get("rules", raw)

        rules = []
        existing_rules = {
            r.rule_text for r in self.memory.symbolic_rules.get_all_rules()
        }
        for item in rules_list:
            if isinstance(item, dict) and item.get("rule_text") not in existing_rules:
                rules.append(SymbolicRuleORM(**item))
            else:
                self.logger.log(
                    "DuplicateSymbolicRuleSkipped", {"rule_text": item.get("rule_text")}
                )
        return rules

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
            if rule.goal_category != goal.get("focus_area"):
                return False
        return True

    @staticmethod
    def compute_context_hash(context_dict: dict) -> str:
        canonical_str = json.dumps(context_dict, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
