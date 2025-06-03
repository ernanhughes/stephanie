# co_ai/agents/rule_tuner_agent.py

from collections import defaultdict

from co_ai.agents import BaseAgent
from co_ai.constants import GOAL
from co_ai.models import SymbolicRuleORM


class RuleTunerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.min_applications = cfg.get("min_applications", 2)
        self.min_score_delta = cfg.get("min_score_delta", 0.15)
        self.enable_rule_cloning = cfg.get("enable_rule_cloning", True)

    async def run(self, context: dict) -> dict:
        # Step 1: Retrieve all symbolic rules
        rules = self.memory.symbolic_rules.get_all()
        suggestions = []

        for rule in rules:
            applications = self.memory.rule_effects.get_by_rule(rule.id)
            self.logger.log(
                "RuleApplicationCount",
                {
                    "rule_id": rule.id,
                    "count": len(applications),
                },
            )
            if len(applications) < self.min_applications:
                continue

            scores = [a.delta_score for a in applications if a.delta_score is not None]
            if not len(scores):
                continue
            avg_score = sum(scores) / len(scores)

            # Step 2: Compare with others in the same group
            comparison_set = self.memory.rule_effects.get_by_context_hash(rule.context_hash)
            baseline_scores = [a.delta_score for a in comparison_set if a.rule_id != rule.id and a.delta_score is not None]

            if len(baseline_scores) < self.min_applications:
                continue

            baseline_avg = sum(baseline_scores) / len(baseline_scores)
            delta = avg_score - baseline_avg

            self.logger.log("RulePerformanceComparison", {
                "rule_id": rule.id,
                "avg_score": avg_score,
                "baseline": baseline_avg,
                "delta": delta,
            })

            # Step 3: If rule underperforms, clone with tweaks
            if self.enable_rule_cloning and delta < -self.min_score_delta:
                new_rule = self.mutate_rule(rule)
                self.memory.symbolic_rules.insert(new_rule)
                suggestions.append(new_rule.to_dict())
                self.logger.log("RuleClonedDueToPoorPerformance", new_rule.to_dict())

        context["rule_suggestions"] = suggestions
        return context

    def mutate_rule(self, rule: SymbolicRuleORM) -> SymbolicRuleORM:
        new_attrs = dict(rule.attributes)
        # Example tweak: toggle model if present
        if "model.name" in new_attrs:
            old_model = new_attrs["model.name"]
            new_model = "ollama/phi3" if "qwen" in old_model else "ollama/qwen3"
            new_attrs["model.name"] = new_model

        return SymbolicRuleORM(
            source="rule_tuner",
            target=rule.target,
            filter=rule.filter,
            attributes=new_attrs,
            context_hash=SymbolicRuleORM.compute_context_hash(rule.filter, new_attrs),
            agent_name=rule.agent_name,
            goal_type=rule.goal_type,
            goal_category=rule.goal_category,
            difficulty=rule.difficulty,
        )
