# stephanie/agents/rule_generator.py
import statistics
from collections import defaultdict

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.models import (EvaluationORM, PipelineRunORM,
                              RuleApplicationORM, SymbolicRuleORM)


class RuleGeneratorAgent(BaseAgent):
    def __init__(self, *args, min_value_threshold=7.5, min_repeat_count=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_value_threshold = min_value_threshold
        self.min_repeat_count = min_repeat_count

    async def run(self, context: dict) -> dict:
        self.logger.log("RuleGeneratorStart", {"run_id": context.get(PIPELINE_RUN_ID)})
        new_rules = []

        # Step 1: Get high-scoring runs without rule applications
        high_scores = self._get_high_performance_runs()
        grouped = self._group_by_context_signature(high_scores)

        for sig, entries in grouped.items():
            if len(entries) < self.min_repeat_count:
                continue

            # Check if a rule already exists for this context
            if self.memory.symbolic_rules.exists_by_signature(sig):
                continue

            # Step 2a: Heuristic-based rule suggestion
            rule = self._create_rule_from_signature(sig)
            if rule:
                self.memory.symbolic_rules.insert(rule)
                self.logger.log("HeuristicRuleGenerated", rule.to_dict())
                new_rules.append(rule.to_dict())
            else:
                # Step 2b: LLM fallback
                prompt = self._build_llm_prompt(entries)
                response = self.call_llm(prompt, context)
                self.logger.log("LLMGeneratedRule", {"response": response})
                # Optionally parse/validate this into a SymbolicRuleORM

        context["generated_rules"] = new_rules
        self.logger.log("RuleGeneratorEnd", {"generated_count": len(new_rules)})
        return context

    def _get_high_performance_runs(self):
        scores = (
            self.memory.session.query(EvaluationORM)
            .filter(EvaluationORM.score >= self.min_value_threshold)
            .all()
        )
        runs = []
        for score in scores:
            rule_app = (
                self.memory.session.query(RuleApplicationORM)
                .filter_by(hypothesis_id=score.hypothesis_id)
                .first()
            )
            if rule_app:
                continue  # Skip if rule already applied
            run = self.memory.session.get(PipelineRunORM, score.pipeline_run_id)
            if run:
                runs.append((score, run))
        return runs

    def _group_by_context_signature(self, scored_runs):
        grouped = defaultdict(list)
        for score, run in scored_runs:
            sig = self._make_signature(run.config)
            grouped[sig].append((score, run))
        return grouped

    def _make_signature(self, config: dict) -> str:
        # Could hash or stringify parts of the config, e.g. model + agent + goal
        model = config.get("model", {}).get("name")
        agent = config.get("agent")
        goal_type = config.get("goal", {}).get("goal_type")
        return f"{model}::{agent}::{goal_type}"

    def _create_rule_from_signature(self, sig: str) -> SymbolicRuleORM:
        try:
            model, agent, goal_type = sig.split("::")
            return SymbolicRuleORM(
                source="rule_generator",
                target="agent",
                filter={"goal_type": goal_type},
                attributes={"model.name": model},
                agent_name=agent,
                context_hash=SymbolicRuleORM.compute_context_hash(
                    {"goal_type": goal_type}, {"model.name": model}
                ),
            )
        except Exception as e:
            self.logger.log("SignatureParseError", {"sig": sig, "error": str(e)})
            return None

    def _build_llm_prompt(self, entries: list) -> str:
        examples = "\n\n".join(
            f"Goal: {e[1].config.get('goal', {}).get('goal_text')}\n"
            f"Agent: {e[1].config.get('agent')}\n"
            f"Model: {e[1].config.get('model', {}).get('name')}\n"
            f"Score: {e[0].score}"
            for e in entries[:3]
        )
        return f"""You are a symbolic AI pipeline optimizer.
Given the following successful pipeline configurations with high scores, suggest a symbolic rule that could be applied to future similar tasks.

Examples:
{examples}

Return a YAML snippet that defines a rule with `target`, `filter`, and `attributes`.
"""
