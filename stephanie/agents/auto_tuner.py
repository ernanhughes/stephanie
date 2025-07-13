# stephanie/agents/auto_tuner.py
from datetime import datetime

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.rule_effect_analyzer import RuleEffectAnalyzer
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.models import SymbolicRuleORM


class AutoTunerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.eval_threshold = cfg.get("eval_threshold", 50)
        self.max_rules = cfg.get("max_rules_to_consider", 10)
        self.tune_mode = cfg.get("tune_mode", False)  # Only write changes if True

    async def run(self, context: dict) -> dict:
        self.logger.log("AutoTunerStart", {"tune_mode": self.tune_mode})

        analyzer = RuleEffectAnalyzer(self.memory.session, logger=self.logger)
        summary = analyzer.analyze(context.get(PIPELINE_RUN_ID))

        underperforming_rules = [
            (rule_id, data)
            for rule_id, data in summary.items()
            if data.get("avg_score", 100) < self.eval_threshold
        ]
        underperforming_rules = sorted(
            underperforming_rules, key=lambda x: x[1]["avg_score"]
        )[: self.max_rules]

        for rule_id, data in underperforming_rules:
            rule = self.memory.symbolic_rules.get_by_id(rule_id)
            if not rule:
                continue

            self.logger.log(
                "RuleUnderperforming",
                {
                    "rule_id": rule_id,
                    "avg_score": data["avg_score"],
                    "context_hash": rule.context_hash,
                    "attributes": rule.attributes,
                },
            )

            suggestions = self.suggest_rule_edits(rule, data)
            for suggestion in suggestions:
                self.logger.log(
                    "AutoRuleSuggestion",
                    {
                        "rule_id": rule_id,
                        "suggested_attributes": suggestion,
                        "reason": "AutoTuner based on score analysis",
                    },
                )

                if self.tune_mode:
                    new_rule = SymbolicRuleORM(
                        target=rule.target,
                        filter=rule.filter,
                        attributes=suggestion,
                        source="auto_tuner",
                        created_at=datetime.utcnow(),
                        context_hash=SymbolicRuleORM.compute_context_hash(
                            suggestion, rule.filter
                        ),
                        description=f"Auto-tuned from rule {rule_id}",
                    )
                    self.memory.symbolic_rules.insert(new_rule)
                    self.logger.log("AutoRuleInserted", {"new_rule_id": new_rule.id})

        self.logger.log("AutoTunerEnd", {"rules_checked": len(underperforming_rules)})
        return context

    def suggest_rule_edits(self, rule: SymbolicRuleORM, data: dict) -> list[dict]:
        """
        Heuristic placeholder: try tweaking `temperature`, `max_tokens`, `adapter`, etc.
        Could be replaced by LLM-based or learned tuner later.
        """
        original = rule.attributes or {}

        candidates = []

        if "temperature" in original:
            try:
                temp = float(original["temperature"])
                new_temp = round(min(temp + 0.2, 1.0), 2)
                candidates.append({**original, "temperature": new_temp})
            except:
                pass

        if "adapter" in original:
            candidates.append({**original, "adapter": "default"})

        if "max_tokens" in original:
            try:
                new_tokens = max(int(original["max_tokens"]) - 100, 100)
                candidates.append({**original, "max_tokens": new_tokens})
            except:
                pass

        # Default fallback: add a hint flag
        candidates.append({**original, "hint": "reviewed by tuner"})

        return candidates
