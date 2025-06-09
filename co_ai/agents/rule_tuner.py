from collections import defaultdict

from co_ai.agents.base import BaseAgent
from co_ai.analysis.rule_effect_analyzer import RuleEffectAnalyzer
from co_ai.constants import GOAL, PIPELINE_RUN_ID
from co_ai.memory.symbolic_rule_store import SymbolicRuleStore
from co_ai.models import (EvaluationORM, PipelineRunORM, RuleApplicationORM,
                          SymbolicRuleORM)
from co_ai.rules import RuleTuner


class RuleTunerAgent(BaseAgent):
    """
    Analyzes score dimensions from previous pipeline run and adjusts symbolic rule priorities or parameters.
    Also generates new symbolic rules for repeated high-performing configurations.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.score_target = cfg.get("score_target", "correctness")  # could be 'overall', 'clarity', etc.
        self.rule_store = SymbolicRuleStore(memory=self.memory, logger=self.logger)
        self.rule_tuner = RuleTuner(memory=self.memory, logger=self.logger)
        self.min_score_threshold = cfg.get("min_score_threshold", 7.5)
        self.min_repeat_count = cfg.get("min_repeat_count", 2)

    async def run(self, context: dict) -> dict:
        run_id = context.get(PIPELINE_RUN_ID)
        goal = context.get(GOAL)

        self.logger.log("RuleTunerAgentStart", {"run_id": run_id, "goal_id": goal.get("id")})

        # Analyze which rules were effective
        analyzer = RuleEffectAnalyzer(session=self.memory.session, logger=self.logger)
        effects = analyzer.analyze(run_id)

        # Score target: e.g. maximize 'correctness' or 'reward'
        best_rules = [rid for rid, data in effects.items() if self.score_target in data.get("dimensions", {})]

        self.logger.log("BestRulesIdentified", {
            "target": self.score_target,
            "count": len(best_rules),
            "examples": best_rules[:5]
        })

        # Tune rule parameters or priorities based on dimension performance
        for rule_id in best_rules:
            result = self.rule_tuner.increase_priority(rule_id)
            self.logger.log("RulePriorityIncreased", {"rule_id": rule_id, "new_priority": result})

        context["rule_tuning"] = {
            "target": self.score_target,
            "top_rules": best_rules
        }

        # Auto-generate rules from high-performing runs without rules
        new_rules = self._generate_rules_from_high_scores()
        context["generated_rules"] = new_rules

        self.logger.log("RuleTunerAgentEnd", {"goal_id": goal.get("id"), "run_id": run_id})
        return context

    def _generate_rules_from_high_scores(self):
        scores = self.memory.session.query(EvaluationORM).filter(EvaluationORM.score >= self.min_score_threshold).all()
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

        grouped = defaultdict(list)
        for score, run in runs:
            sig = self._make_signature(run.config)
            grouped[sig].append((score, run))

        new_rules = []
        for sig, entries in grouped.items():
            if len(entries) < self.min_repeat_count:
                continue

            if self.memory.symbolic_rules.exists_by_signature(sig):
                continue

            rule = self._create_rule_from_signature(sig)
            if rule:
                self.memory.symbolic_rules.insert(rule)
                self.logger.log("HeuristicRuleGenerated", rule.to_dict())
                new_rules.append(rule.to_dict())

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
                )
            )
        except Exception as e:
            self.logger.log("SignatureParseError", {"sig": sig, "error": str(e)})
            return None
