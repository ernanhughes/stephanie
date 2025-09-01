import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from stephanie.memory.symbolic_rule_store import SymbolicRuleORM


class SymbolicRuleApplier:
    """
    Core component for applying symbolic rules to modify agent behavior, prompt configurations,
    and pipeline structure at runtime. Acts as the Meta-Controller from the symbolic learning paper.
    
    Key responsibilities:
    - Loading symbolic rules from YAML files and database
    - Matching rules to current execution context
    - Applying rule-based modifications to agents, prompts, and pipelines
    - Tracking rule applications for later analysis and optimization
    """

    def __init__(self, cfg: Dict, memory: Any, logger: Any):
        """
        Initialize the symbolic rule applier.
        
        Args:
            cfg: Configuration dictionary containing symbolic learning settings
            memory: Memory component for accessing and storing rules and applications
            logger: Logger for tracking rule applications and modifications
        """
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.enabled = cfg.get("symbolic", {}).get("enabled", False)
        self._rules = self._load_rules() if self.enabled else []

    @property
    def rules(self) -> list:
        """Get all loaded symbolic rules."""
        return self._rules

    def apply(self, context: dict) -> dict:
        """
        Apply pipeline-level symbolic rules to modify the execution pipeline.
        
        This method can restructure the entire pipeline sequence based on symbolic directives.
        Typically invoked during pipeline planning phase.
        
        Args:
            context: Current execution context containing goal, pipeline, and run information
            
        Returns:
            Updated context with potential pipeline modifications
        """
        if not self.enabled:
            return context

        goal = context.get("goal", {})
        current_pipeline = context.get("pipeline", [])

        # Find rules matching current goal metadata
        matching_rules = [r for r in self.rules if self._matches_metadata(r, goal)]

        if not matching_rules:
            self.logger.log("NoSymbolicRulesApplied", {"goal_id": goal.get("id")})
            return context

        self.logger.log("SymbolicRulesFound", {"count": len(matching_rules)})

        # Apply each matching rule
        for rule in matching_rules:
            # Handle pipeline restructuring rules
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

            # Add strategy hints for lookahead rules
            if rule.source == "lookahead" and rule.goal_type:
                context["symbolic_hint"] = f"use_{rule.goal_type.lower()}_strategy"

        return context

    def apply_to_agent(self, cfg: Dict, context: Dict) -> Dict:
        """
        Apply agent-level symbolic rules to modify agent configuration.
        
        This method is typically called before agent instantiation to override
        default parameters like model selection, adapters, or temperature.
        
        Args:
            cfg: Current agent configuration to be modified
            context: Execution context containing goal and run information
            
        Returns:
            Updated agent configuration with rule-based modifications
        """
        if not self.enabled:
            return cfg

        goal = context.get("goal", {})
        pipeline_run_id = context.get("pipeline_run_id")
        agent_name = cfg.get("name")

        # Find rules targeting this specific agent and matching goal context
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

        # Apply each matching rule to the agent configuration
        for rule in matching_rules:
            # Apply new-style attributes (structured YAML format)
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

            # Apply legacy rule_text format (comma-separated for backward compatibility)
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

            # Record rule application for later analysis and scoring
            self.memory.rule_effects.insert(
                goal_id=goal.get("id"),
                agent_name=agent_name,
                rule_id=rule.id,
                pipeline_run_id=pipeline_run_id,
                details=rule.to_dict(),
                stage_details=cfg,  # Snapshot of configuration after rule application
            )

        return cfg

    def apply_prompt_rules(
        self, agent_name: str, prompt_cfg: dict, context: dict
    ) -> dict:
        """
        Apply prompt-level symbolic rules to modify prompt configuration.
        
        This method is typically called before prompt generation to override
        template selection, formatting options, or other prompt parameters.
        
        Args:
            agent_name: Name of the agent using the prompt
            prompt_cfg: Current prompt configuration to be modified
            context: Execution context containing goal and run information
            
        Returns:
            Updated prompt configuration with rule-based modifications
        """
        goal = context.get("goal", {})
        applicable_rules = [
            rule
            for rule in self.rules
            if rule.agent_name == agent_name
            # and self._matches_filter(rule.filter, goal)  # Optional filter matching
        ]

        if not applicable_rules:
            self.logger.log("NoPromptRulesFound", {"agent": agent_name})
            return prompt_cfg

        # Apply each matching rule to the prompt configuration
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

            # Record rule application for later analysis
            self.memory.rule_effects.insert(
                rule_id=rule.id,
                goal_id=goal.get("id"),
                pipeline_run_id=context.get("pipeline_run_id"),
                details=prompt_cfg,
            )

        return prompt_cfg

    def set_nested(self, cfg: dict, dotted_key: str, value: Any):
        """
        Set a nested configuration value using dot notation.
        
        Example: set_nested(cfg, "model.temperature", 0.7)
        
        Args:
            cfg: Configuration dictionary to modify
            dotted_key: Key in dot notation (e.g., "model.name")
            value: Value to set at the specified nested key
        """
        keys = dotted_key.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def apply_to_prompt(self, cfg: Dict, context: Dict) -> Dict:
        """
        Alternative method for applying prompt-level rules (maintains backward compatibility).
        
        Args:
            cfg: Prompt configuration to modify
            context: Execution context containing goal and run information
            
        Returns:
            Updated prompt configuration
        """
        if not self.enabled:
            return cfg

        goal = context.get("goal", {})
        pipeline_run_id = context.get("pipeline_run_id")
        prompt_name = cfg.get("prompt_key", "unknown_prompt")

        # Find rules targeting prompts and matching current context
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

        # Apply each matching rule
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

            # Track rule application
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
        """
        Check if target object matches all conditions in the filter dictionary.
        
        Supports both single values and lists of acceptable values.
        
        Args:
            filter_dict: Rule filter conditions (e.g., {"goal_type": "research"})
            target_obj: Object to check against filters (e.g., goal metadata)
            
        Returns:
            True if all filter conditions are satisfied, False otherwise
        """
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
        """Track pipeline stage execution for later analysis."""
        self.memory.symbolic_rules.track_pipeline_stage(stage_dict, context)

    def get_nested_value(d: dict, key_path: str) -> Any:
        """
        Get a nested value from a dictionary using dot notation.
        
        Args:
            d: Dictionary to search
            key_path: Dot-separated path to the value (e.g., "model.name")
            
        Returns:
            Value at the specified path or None if not found
        """
        keys = key_path.split(".")
        for key in keys:
            d = d.get(key, {})
        return d if d else None

    def set_nested_value(d: dict, key_path: str, value: Any):
        """
        Set a nested value in a dictionary using dot notation.
        
        Args:
            d: Dictionary to modify
            key_path: Dot-separated path where value should be set
            value: Value to set at the specified path
        """
        keys = key_path.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _load_rules(self) -> List[SymbolicRuleORM]:
        """
        Load symbolic rules from both YAML files and database.
        
        Returns:
            List of all loaded symbolic rules
        """
        rules = []
        symbolic_dict = self.cfg.get("symbolic", {})
        
        # Load from YAML file if specified
        if symbolic_dict.get("rules_file"):
            rules += self._load_rules_from_yaml(symbolic_dict.get("rules_file"))
            
        # Load from database if enabled
        if symbolic_dict.get("enable_db_rules", True):
            rules += self.memory.symbolic_rules.get_all_rules()
            
        return rules

    def _load_rules_from_yaml(self, path: str) -> List[SymbolicRuleORM]:
        """
        Load symbolic rules from a YAML configuration file.
        
        Args:
            path: Path to the YAML file containing rule definitions
            
        Returns:
            List of rules loaded from the YAML file
        """
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

        # Convert YAML entries to SymbolicRuleORM objects
        for item in rules_list:
            if isinstance(item, dict) and item.get("rule_text") not in existing_rules:
                rules.append(SymbolicRuleORM(**item))
            else:
                self.logger.log(
                    "DuplicateSymbolicRuleSkipped", {"rule_text": item.get("rule_text")}
                )
                
        return rules

    def _matches_metadata(self, rule: SymbolicRuleORM, goal: Dict[str, Any]) -> bool:
        """
        Check if a rule matches the goal metadata.
        
        Args:
            rule: Symbolic rule to check
            goal: Goal metadata to match against
            
        Returns:
            True if rule matches all relevant goal metadata, False otherwise
        """
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
        """
        Compute a deterministic hash for a context dictionary.
        
        Used for deduplication and grouping similar rule applications.
        
        Args:
            context_dict: Context dictionary to hash
            
        Returns:
            SHA256 hash of the canonicalized context
        """
        canonical_str = json.dumps(context_dict, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()