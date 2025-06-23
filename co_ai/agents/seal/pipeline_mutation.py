import copy
import datetime

import yaml
from omegaconf import OmegaConf

from co_ai.agents.base_agent import BaseAgent
from co_ai.models import SymbolicRuleORM
from co_ai.registry.pipeline import PipelineRegistry
from co_ai.rules.rule_options_config import RuleOptionsConfig
from co_ai.rules.rule_tuner import RuleTuner
from co_ai.supervisor import Supervisor


class PipelineMutationAgent(BaseAgent):
    """
    Combines symbolic rule mutation with pipeline configuration mutation.
    Generates both types of mutations, applies them, evaluates outcomes,
    and logs improvements for future learning.
    """

    def __init__(
        self,
        cfg,
        memory,
        logger,
        full_cfg=None,
    ):
        super().__init__(cfg, memory, logger)
        self.full_cfg = full_cfg
        self.target_agent = cfg.get("target_agent", "default")
        self.mutation_prompt_template = cfg["rule_mutation_prompt"]
        self.max_runs = cfg.get("max_runs", 5)


        # Load base pipeline
        self.base_pipeline_key = cfg.get("base_pipeline", "minimal")
        self.pipeline_registry_path = cfg.get("pipeline_registry", "config/registry/pipeline_registry.yaml")
        self.pipeline_registry = PipelineRegistry(self.pipeline_registry_path)

        self.rule_options_file = cfg.get("mutation_rule_options", "config/rules/pipeline_mutation_options.yaml")
        self.options_config = RuleOptionsConfig.from_yaml(self.rule_options_file)
        self.rule_tuner = RuleTuner(memory, logger)

        self.logger.log(
            "PipelineMutationAgentInitialized",
            {"conf": self.cfg}
        )

    async def run(self, context: dict) -> dict:
        # Step 1: Generate pipeline config mutations
        pipeline_def = self.pipeline_registry.get_pipeline(self.base_pipeline_key)
        if not pipeline_def:
            self.logger.log("PipelineNotFound", {"pipeline": self.base_pipeline_key})
            context["status"] = "pipeline_not_found"
            return context

        _, pipeline = self._generate_pipeline_mutations(self.base_pipeline_key, context) 


        # Step 2: Generate symbolic rule mutations
        applicable_rules = self._get_applicable_rules(pipeline)
        symbolic_mutations = []
        for rule in applicable_rules:
            symbolic_mutations.extend(self._generate_rule_mutations(rule, context))

        # Step 3: Apply and evaluate symbolic mutations
        symbolic_results = await self._apply_and_evaluate(symbolic_mutations, context)

        pipeline_to_mutate_def = self.pipeline_registry.get_pipeline(pipeline)

        # Step 4: Apply and evaluate pipeline mutations
        pipeline_results = await self._apply_pipeline_mutations(pipeline_to_mutate_def, symbolic_results, context)

        # Step 5: Log all results
        context["mutated_symbolic_rules"] = [r.to_dict() for r in symbolic_results]
        context["mutated_pipeline_runs"] = pipeline_results
        context["total_mutations_run"] = len(symbolic_results) + len(pipeline_results)

        return context

    def _get_applicable_rules(self, pipeline_name: str) -> list:
        """Get all relevant symbolic First you need to finish this for all agents in a given pipeline."""
        pipeline_def = self.pipeline_registry.get_pipeline(pipeline_name)
        agent_names = {stage.get("name") for stage in pipeline_def if "name" in stage}

        # Filter rules where the rule's agent matches any in the pipeline
        return [
            r for r in self.memory.symbolic_rules.get_all()
            if r.agent_name in agent_names
        ]

    def _generate_rule_mutations(self, rule: SymbolicRuleORM, context: dict) -> list[dict]:
        """Use LLM to generate one or more valid mutations for this rule."""
        current_attrs = rule.attributes or {}
        available_options = self.options_config.get_options_for(rule.agent_name)
        recent_perf = self.memory.rule_effects.get_recent_performance(rule.id)

        merged = {
            "current_attributes": current_attrs,
            "available_options": available_options,
            "recent_performance": recent_perf,
            **context
        }

        prompt = self.prompt_loader.from_file(self.mutation_prompt_template, self.cfg, merged)
        response = self.call_llm(prompt, context)
        parsed = RuleTuner.parse_mutation_response(response)

        if not parsed.get("attribute") or not parsed.get("new_value"):
            self.logger.log("MutationParseError", {"rule_id": rule.id, "response": response})
            return []

        attr = parsed["attribute"]
        new_val = parsed["new_value"]

        if not self.options_config.is_valid_change(rule.agent_name, attr, new_val):
            self.logger.log("InvalidRuleMutation", {"rule_id": rule.id, "attribute": attr, "value": new_val})
            return []

        if self.memory.symbolic_rules.exists_similar(rule, attr, new_val):
            self.logger.log("RuleMutationDuplicateSkipped", {"rule_id": rule.id, "attribute": attr, "value": new_val})
            return []

        mutated_attrs = dict(current_attrs)
        mutated_attrs[attr] = new_val

        new_rule = SymbolicRuleORM(
            target="agent",
            agent_name=rule.agent_name,
            goal_type=rule.goal_type,
            goal_category=rule.goal_category,
            difficulty=rule.difficulty,
            attributes=mutated_attrs,
            source="mutation",
        )
        self.memory.symbolic_rules.insert(new_rule)
        self.logger.log("RuleMutat I ionApplied", {"original_rule_id": rule.id, "new_rule": new_rule.to_dict()})
        return [new_rule]

    def _generate_pipeline_mutations(self, pipeline_name, context):
        """Generate pipeline config mutations using LLM guidance"""

        merged_context = {
            # From pipeline definition
            "current_pipeline_name": pipeline_name,
            "current_pipeline_description": self.pipeline_registry.get_description(pipeline_name),
            "current_pipeline": self.pipeline_registry.get_pipeline(pipeline_name),  # handles if it's a full pipeline block

            # From context (goal and performance)
            "goal_text": context.get("goal", {}).get("goal_text", "Improve pipeline performance"),
            "goal_id": context.get("goal", {}).get("id"),
            #TODO
            # "recent_performance": self.memory.rule_effects.get_recent_performance_summary(),

            # Optionally, inject available options for better prompting
            "available_pipelines": self.pipeline_registry.list_variants_with_descriptions(),  # e.g., [{"name": ..., "description": ...}, ...]

            # Pass original context for compatibility
            **context,
        }

        prompt = self.prompt_loader.from_file("pipeline", self.cfg, merged_context)
        response = self.call_llm(prompt, context)
        rationale, pipeline  = self._parse_pipeline_mutation(response)

        if not pipeline:
            self.logger.log("PipelineMutationParseError", {"response": response})
            return []

        return rationale, pipeline

    def _parse_pipeline_mutation(self, response: str):
        import re
        """Parse LLM response into a pipeline mutation"""
        pattern = r"""
        (?:[*#`]*\s*)?            # Optional formatting characters before the header
        rationale\s*:             # Match the word "rationale:"
        \s*(?P<rationale>.*?)     # Capture rationale content non-greedily
        (?:\n|\r|\r\n)+           # Match the newline(s) separating the two blocks
        (?:[*#`]*\s*)?            # Optional formatting characters before the second header
        pipeline\s*:\s*           # Match "pipeline:"
        (?P<pipeline>\w+)         # Capture pipeline name
        """
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL | re.VERBOSE)

        if match:
            rationale = match.group("rationale").strip()
            pipeline = match.group("pipeline").strip()
        return rationale, pipeline    
            
    async def _apply_and_evaluate(self, mutations: list[SymbolicRuleORM], context: dict) -> list[SymbolicRuleORM]:
        """Apply each symbolic mutation and evaluate its effect."""
        results = []

        for rule in mutations:
            new_config = self._apply_symbolic_rule(rule)
            mutated_context = self._update_context_with_config(context, new_config)

            supervisor = Supervisor(self.full_cfg, memory=self.memory, logger=self.logger)
            result = await supervisor.run_pipeline_config(mutated_context)

            score = self._evaluate_result(result)
            self._log_evaluation(rule, score)

            if score > 0.5:
                results.append(rule)

        return results

    def _apply_symbolic_rule(self, rule: SymbolicRuleORM):
        """Apply symbolic rule to config"""
        # You could do deeper merging here based on agent name
        return {f"{rule.agent_name}.config": rule.attributes}

    def _update_context_with_config(self, context, config_update):
        """Merge symbolic config into context"""
        ctx_copy = copy.deepcopy(context)
        ctx_copy.update(config_update)
        return ctx_copy

    async def _apply_pipeline_mutations(self, pipeline_def, mutations: list, context: dict) -> list:
        """Apply pipeline mutations and run through supervisor"""
        results = []

        for i, mutation in enumerate(mutations):
            if i >= self.max_runs:
                self.logger.log("PipelineMutationLimitReached", {"limit": self.max_runs})
                break

            mutated_pipeline = self.apply_mutation(pipeline_def, mutation)
            mutated_cfg = self.inject_pipeline_config(mutated_pipeline, tag=f"mutated_{i}")

            full_mutated_cfg = OmegaConf.merge(mutated_cfg, self.full_cfg)
            supervisor = Supervisor(full_mutated_cfg, memory=self.memory, logger=self.logger)

            try:
                mutated_run = await supervisor.run_pipeline_config(context)
                summary = self.summarize(mutated_run)
                self.logger.log("PipelineMutationRun", {"mutation": mutation, "summary": summary})
                results.append({"mutation": mutation, "result": mutated_run})
            except Exception as e:
                self.logger.log("PipelineMutationError", {"mutation": mutation, "error": str(e)})

        return results

    def apply_mutation(self, pipeline_cfg: list, mutation: dict) -> list:
        """Apply a single mutation to a deep copy of the pipeline config."""
        mutated = copy.deepcopy(pipeline_cfg)
        for key, value in mutation.items():
            keys = key.split(".")
            target = mutated
            for k in keys[:-1]:
                target = target.setdefault(k, {})
            target[keys[-1]] = value
        return mutated

    def inject_pipeline_config(self, pipeline_def, tag="mutated") -> OmegaConf:
        """Replace pipeline stages in full config"""
        full_cfg = OmegaConf.to_container(self.full_cfg, resolve=True)
        full_cfg["pipeline"]["tag"] = tag
        full_cfg["pipeline"]["stages"] = pipeline_def
        full_cfg["agents"] = {stage["name"]: stage for stage in pipeline_def}
        return OmegaConf.create(full_cfg)

    def _evaluate_result(self, result: dict) -> float:
        """Score mutation outcome using MRQScorer or other scorer"""
        score = result.get("best_score", 0.0)
        return score

    def _log_evaluation(self, rule: SymbolicRuleORM, score: float):
        """Log mutation and evaluation result"""
        self.memory.scorer.score_db.append({
            "rule_id": rule.id,
            "score": score,
            "timestamp": datetime.now(),
        })

    def summarize(self, result: dict) -> dict:
        """Return short summary for logging"""
        return {
            "goal_id": result.get("goal", {}).get("id"),
            "best_score": result.get("best_score"),
            "selected_hypothesis": result.get("selected", {}).get("text", "")[:50],
        }

    def _load_pipeline_registry(self):
        with open(self.pipeline_registry_path, "r") as f:
            return yaml.safe_load(f)

