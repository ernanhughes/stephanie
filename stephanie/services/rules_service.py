# stephanie/services/rules_service.py
"""
Symbolic Rules Service Module

Core service for applying symbolic rules to modify agent behavior, prompt configurations,
and pipeline structure at runtime. Acts as the Meta-Controller from the symbolic learning paper.

This service implements the symbolic learning component of the Stephanie AI system,
allowing for runtime modifications of agent behavior based on learned rules from
previous executions. It supports loading rules from both YAML files and a database,
and applies them to various components of the system including pipelines, agents, and prompts.

Key Features:
- Rule loading from multiple sources (YAML files and database)
- Context-aware rule matching and application
- Multi-level rule application (pipeline, agent, prompt)
- Rule effect tracking for later analysis and optimization
- Health monitoring and statistics collection
"""

from __future__ import annotations

import datetime
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import yaml

from stephanie.memory.symbolic_rule_store import SymbolicRuleORM
from stephanie.services.service_protocol import Service


class RulesService(Service):
    """
    Core component for applying symbolic rules to modify agent behavior, prompt configurations,
    and pipeline structure at runtime. Acts as the Meta-Controller from the symbolic learning paper.
    
    This service enables dynamic adaptation of the AI system based on learned rules that capture
    successful patterns from previous executions. Rules can modify pipeline structure, agent
    configuration, and prompt parameters based on the current execution context.
    
    Key responsibilities:
    - Loading symbolic rules from YAML files and database
    - Matching rules to current execution context
    - Applying rule-based modifications to agents, prompts, and pipelines
    - Tracking rule applications for later analysis and optimization
    
    Attributes:
        cfg (Dict): Configuration dictionary
        memory: Reference to the memory service
        logger: Reference to the logging service
        enabled (bool): Whether symbolic learning is enabled
        _rules (List[SymbolicRuleORM]): Loaded symbolic rules
        _rules_loaded (bool): Whether rules have been loaded
        _last_loaded (datetime): Timestamp of last rule load
        _rule_stats (Dict): Statistics about rule loading
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
        self._rules: List[SymbolicRuleORM] = []
        self._rules_loaded = False
        self._last_loaded = None
        self._rule_stats = {
            "total_rules": 0,
            "rules_from_yaml": 0,
            "rules_from_db": 0,
            "last_load_time": None,
            "load_duration": 0.0
        }

    @property
    def name(self) -> str:
        """Return the service name for identification."""
        return "rules"
    
    def initialize(self, **kwargs) -> None:
        """
        Initialize rule service by loading rules if enabled.
        
        Args:
            **kwargs: Additional initialization parameters
            
        Raises:
            Exception: If rule loading fails and symbolic learning is enabled
        """
        if not self.enabled:
            self.logger.log("RulesServiceDisabled", {
                "reason": "symbolic learning not enabled in config"
            })
            return
            
        start_time = datetime.datetime.now()
        try:
            self._load_rules()
            self._last_loaded = datetime.datetime.now()
            self._rule_stats["load_duration"] = (self._last_loaded - start_time).total_seconds()
            self.logger.log("RulesServiceInitialized", {
                "total_rules": len(self._rules),
                "rules_from_yaml": self._rule_stats["rules_from_yaml"],
                "rules_from_db": self._rule_stats["rules_from_db"],
                "load_time": self._rule_stats["load_duration"]
            })
        except Exception as e:
            self.logger.log("RulesServiceInitError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Return health status and metrics specific to rule service.
        
        Returns:
            Dictionary containing health information with keys:
            - status: Overall service status ("healthy" or "disabled")
            - enabled: Whether the service is enabled
            - rules_loaded: Whether rules have been loaded
            - total_rules: Total number of loaded rules
            - rules_from_yaml: Number of rules loaded from YAML files
            - rules_from_db: Number of rules loaded from database
            - last_loaded: Timestamp of last rule load
            - load_duration: Duration of last rule load in seconds
        """
        return {
            "status": "healthy" if self.enabled and self._rules_loaded else "disabled",
            "enabled": self.enabled,
            "rules_loaded": self._rules_loaded,
            "total_rules": len(self._rules),
            "rules_from_yaml": self._rule_stats["rules_from_yaml"],
            "rules_from_db": self._rule_stats["rules_from_db"],
            "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
            "load_duration": self._rule_stats["load_duration"]
        }
    
    def shutdown(self) -> None:
        """Cleanly shut down the service and release all resources."""
        self._rules = []
        self._rules_loaded = False
        self.logger.log("RulesServiceShutdown", {
            "status": "complete",
            "rules_cleared": True
        })

    @property
    def rules(self) -> list:
        """
        Get all loaded symbolic rules, loading them if necessary.
        
        Returns:
            List of loaded symbolic rules or empty list if service is disabled
        """
        if not self.enabled:
            return []
            
        if not self._rules_loaded:
            self.initialize()
            
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
        if not self.enabled:
            return prompt_cfg
            
        goal = context.get("goal", {})
        applicable_rules = [
            rule
            for rule in self.rules
            if rule.agent_name == agent_name and self._matches_metadata(rule, goal)
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
                        "old_value": self.get_nested_value(prompt_cfg, key),
                        "new_value": value,
                        "rule_id": rule.id,
                        "emoji": "🛠️",
                    },
                )
                self.set_nested_value(prompt_cfg, key, value)

            # Record rule application for later analysis
            self.memory.rule_effects.insert(
                rule_id=rule.id,
                goal_id=goal.get("id"),
                pipeline_run_id=context.get("pipeline_run_id"),
                details=prompt_cfg,
            )

        return prompt_cfg

    def set_nested_value(self, cfg: dict, key_path: str, value: Any):
        """
        Set a nested value in a dictionary using dot notation.
        
        Args:
            cfg: Dictionary to modify
            key_path: Dot-separated path where value should be set
            value: Value to set at the specified path
        """
        keys = key_path.split(".")
        for key in keys[:-1]:
            cfg = cfg.setdefault(key, {})
        cfg[keys[-1]] = value

    def get_nested_value(self, cfg: dict, key_path: str) -> Any:
        """
        Get a nested value from a dictionary using dot notation.
        
        Args:
            cfg: Dictionary to search
            key_path: Dot-separated path to the value (e.g., "model.name")
            
        Returns:
            Value at the specified path or None if not found
        """
        keys = key_path.split(".")
        for key in keys:
            cfg = cfg.get(key, {})
            if not isinstance(cfg, dict):
                return cfg
        return None

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
            if r.target == "prompt" and self._matches_metadata(r, goal)
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
            
        # Handle focus_area as alternative to goal_category
        focus_area = goal.get("focus_area") or goal.get("focus_area")
        if focus_area and rule.goal_category and rule.goal_category != focus_area:
            return False
            
        return True

    def _load_rules(self) -> List[SymbolicRuleORM]:
        """
        Load symbolic rules from both YAML files and database.
        
        Returns:
            List of all loaded symbolic rules
            
        Note:
            This method loads rules from both YAML files (if specified in config)
            and from the database (if enabled). It tracks statistics about the
            loading process and handles errors gracefully.
        """
        if self._rules_loaded:
            return self._rules

        start_time = datetime.datetime.now()
        rules = []
        symbolic_dict = self.cfg.get("symbolic", {})
        self._rule_stats["rules_from_yaml"] = 0
        self._rule_stats["rules_from_db"] = 0
        
        # Load from YAML file if specified
        if symbolic_dict.get("rules_file"):
            try:
                yaml_rules = self._load_rules_from_yaml(symbolic_dict.get("rules_file"))
                rules.extend(yaml_rules)
                self._rule_stats["rules_from_yaml"] = len(yaml_rules)
            except Exception as e:
                self.logger.log("SymbolicRuleYAMLError", {
                    "path": symbolic_dict.get("rules_file"),
                    "error": str(e)
                })

        # Load from database if enabled
        if symbolic_dict.get("enable_db_rules", True):
            try:
                db_rules = self.memory.symbolic_rules.get_all_rules()
                rules.extend(db_rules)
                self._rule_stats["rules_from_db"] = len(db_rules)
            except Exception as e:
                self.logger.log("SymbolicRuleDBError", {
                    "error": str(e)
                })

        self._rules = rules
        self._rules_loaded = True
        self._rule_stats["total_rules"] = len(rules)
        self._last_loaded = datetime.datetime.now()
        self._rule_stats["load_duration"] = (self._last_loaded - start_time).total_seconds()
        
        self.logger.log("RulesLoaded", {
            "total": len(rules),
            "from_yaml": self._rule_stats["rules_from_yaml"],
            "from_db": self._rule_stats["rules_from_db"],
            "duration": self._rule_stats["load_duration"]
        })
        
        return rules

    def _load_rules_from_yaml(self, path: str) -> List[SymbolicRuleORM]:
        """
        Load symbolic rules from a YAML configuration file.
        
        Args:
            path: Path to the YAML file containing rule definitions
            
        Returns:
            List of rules loaded from the YAML file
            
        Note:
            This method handles YAML parsing and converts YAML entries to
            SymbolicRuleORM objects. It skips duplicates and malformed entries.
        """
        path = Path(path)
        if not path.exists():
            self.logger.log("SymbolicRuleYAMLNotFound", {"path": str(path)})
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
                
            rules_list = raw.get("rules", [])
            if not rules_list:
                self.logger.log("SymbolicRuleYAMLEmpty", {"path": str(path)})
                return []
                
            rules = []
            existing_rules = {r.rule_text for r in self.memory.symbolic_rules.get_all_rules()}
            
            # Convert YAML entries to SymbolicRuleORM objects
            for item in rules_list:
                if not isinstance(item, dict):
                    continue
                    
                rule_text = item.get("rule_text", "").strip()
                if not rule_text:
                    continue
                    
                # Skip duplicates
                if rule_text in existing_rules:
                    continue
                    
                # Create rule with proper defaults
                rule = SymbolicRuleORM(
                    rule_text=rule_text,
                    agent_name=item.get("agent_name", "all"),
                    target=item.get("target", "pipeline"),
                    attributes=item.get("attributes", {}),
                    filter=item.get("filter", {}),
                    source=item.get("source", "yaml"),
                    goal_id=item.get("goal_id"),
                    goal_type=item.get("goal_type"),
                    goal_category=item.get("goal_category"),
                    difficulty=item.get("difficulty"),
                    priority=item.get("priority", 0),
                    created_at=datetime.datetime.now()
                )
                rules.append(rule)
                
            return rules
            
        except Exception as e:
            self.logger.log("SymbolicRuleYAMLParseError", {
                "path": str(path),
                "error": str(e)
            })
            return []

    @staticmethod
    def compute_context_hash(context_dict: dict) -> str:
        """
        Compute a deterministic hash for a context dictionary.
        
        Used for deduplication and grouping similar rule applications.
        
        Args:
            context_dict: Context dictionary to hash
            
        Returns:
            SHA256 hash of the canonicalized context
            
        Note:
            This method creates a canonical representation of the context by
            sorting keys and normalizing values to ensure consistent hashing
            across different executions.
        """
        if not context_dict:
            return "empty_context"
            
        # Create a canonical representation for hashing
        canonical_dict = {}
        for k, v in sorted(context_dict.items()):
            if isinstance(v, dict):
                canonical_dict[k] = RulesService.compute_context_hash(v)
            elif isinstance(v, list):
                # Sort lists of primitives if possible
                if all(isinstance(x, (str, int, float)) for x in v):
                    canonical_dict[k] = sorted(v)
                else:
                    canonical_dict[k] = [RulesService.compute_context_hash(x) if isinstance(x, dict) else str(x) for x in v]
            else:
                canonical_dict[k] = str(v)
                
        canonical_str = json.dumps(canonical_dict, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()