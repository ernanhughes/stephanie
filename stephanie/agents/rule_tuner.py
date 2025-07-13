# stephanie/agents/rule_tuner.py
from collections import defaultdict

from sqlalchemy.orm import joinedload
from tabulate import tabulate

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.rule_effect_analyzer import RuleEffectAnalyzer
from stephanie.constants import GOAL, PIPELINE_RUN_ID
from stephanie.memory.symbolic_rule_store import SymbolicRuleStore
from stephanie.models import SymbolicRuleORM
from stephanie.rules import RuleTuner
from stephanie.utils.high_score_selector import get_high_scoring_runs


class RuleTunerAgent(BaseAgent):
    """
    Analyzes score dimensions from previous pipeline run and adjusts symbolic rule priorities or parameters.
    Also generates new symbolic rules for repeated high-performing configurations.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.score_target = cfg.get(
            "score_target", "correctness"
        )  # could be 'overall', 'clarity', etc.
        self.rule_store = SymbolicRuleStore(
            session=self.memory.session, logger=self.logger
        )
        self.rule_tuner = RuleTuner(memory=self.memory, logger=self.logger)
        self.min_score_threshold = cfg.get("min_score_threshold", 7.5)
        self.min_repeat_count = cfg.get("min_repeat_count", 2)

    async def run(self, context: dict) -> dict:
        run_id = context.get(PIPELINE_RUN_ID)
        goal = context.get(GOAL)

        self.logger.log(
            "RuleTunerAgentStart", {"run_id": run_id, "goal_id": goal.get("id")}
        )

        # Analyze which rules were effective
        analyzer = RuleEffectAnalyzer(session=self.memory.session, logger=self.logger)
        effects = analyzer.analyze(run_id)

        # Score target: e.g. maximize 'correctness' or 'reward'
        best_rules = [
            rid
            for rid, data in effects.items()
            if self.score_target in data.get("dimensions", {})
        ]

        self.logger.log(
            "BestRulesIdentified",
            {
                "target": self.score_target,
                "count": len(best_rules),
                "examples": best_rules[:5],
            },
        )

        # Tune rule parameters or priorities based on dimension performance
        for rule_id in best_rules:
            result = self.rule_tuner.increase_priority(rule_id)
            self.logger.log(
                "RulePriorityIncreased", {"rule_id": rule_id, "new_priority": result}
            )

        context["rule_tuning"] = {"target": self.score_target, "top_rules": best_rules}

        # Auto-generate rules from high-performing runs without rules
        new_rules = self._generate_rules_from_high_scores()
        context["generated_rules"] = new_rules

        self.logger.log(
            "RuleTunerAgentEnd", {"goal_id": goal.get("id"), "run_id": run_id}
        )
        return context

    def _generate_rules_from_high_scores(self):
        grouped = get_high_scoring_runs(
            session=self.memory.session,
            dimension=self.score_target,
            threshold=self.min_score_threshold,
            min_repeat_count=self.min_repeat_count,
        )

        new_rules = []
        for sig, entries in grouped.items():
            if self.memory.symbolic_rules.exists_by_signature(sig):
                continue

            rule = self._create_rule_from_signature(sig)
            if rule:
                self.memory.symbolic_rules.insert(rule)
                self.logger.log("HeuristicRuleGenerated", rule.to_dict())
                new_rules.append(rule.to_dict())

        if new_rules:
            table = [
                [
                    rule.get("agent_name"),
                    rule.get("attributes", {}).get("model.name"),
                    rule.get("filter", {}).get("goal_type"),
                    rule.get("context_hash", "")[:8],  # short hash
                ]
                for rule in new_rules
            ]

            print("\n📜 New Symbolic Rules Generated:\n")
            print(
                tabulate(
                    table,
                    headers=["Agent", "Model", "Goal Type", "Hash"],
                    tablefmt="fancy_grid",
                )
            )
        else:
            print("\n⚠️  No new symbolic rules were generated.\n")

        return new_rules

    def _make_signature(self, config: dict) -> str:
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
