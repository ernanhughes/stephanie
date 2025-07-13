# stephanie/agents/rule_refiner.py
import statistics

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.models.rule_application import RuleApplicationORM
from stephanie.models.symbolic_rule import SymbolicRuleORM


class RuleRefinerAgent(BaseAgent):
    def __init__(self, *args, min_applications=3, min_score_threshold=6.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_applications = min_applications
        self.min_score_threshold = min_score_threshold

    async def run(self, context: dict) -> dict:
        self.logger.log("RuleRefinerStart", {"run_id": context.get(PIPELINE_RUN_ID)})

        rule_apps = self.memory.session.query(RuleApplicationORM).all()
        grouped = self._group_by_rule(rule_apps)

        for rule_id, applications in grouped.items():
            if len(applications) < self.min_applications:
                continue

            scores = [
                app.result_score for app in applications if app.result_score is not None
            ]
            if not scores:
                continue

            avg_score = statistics.mean(scores)
            if avg_score >= self.min_score_threshold:
                continue  # Only refine low-performing rules

            rule = self.memory.session.query(SymbolicRuleORM).get(rule_id)
            self.logger.log(
                "LowPerformingRuleFound",
                {
                    "rule_id": rule_id,
                    "applications": len(scores),
                    "avg_score": avg_score,
                },
            )

            refinement_prompt = self._build_prompt(rule, scores, applications)
            response = self.call_llm(refinement_prompt, context)
            self.logger.log(
                "RefinementSuggestion",
                {"rule_id": rule_id, "suggestion": response.strip()},
            )

        self.logger.log("RuleRefinerEnd", {"run_id": context.get(PIPELINE_RUN_ID)})
        return context

    def _group_by_rule(self, rule_apps):
        grouped = {}
        for app in rule_apps:
            grouped.setdefault(app.rule_id, []).append(app)
        return grouped

    def _build_prompt(
        self, rule: SymbolicRuleORM, scores: list, applications: list
    ) -> str:
        attributes_str = str(rule.attributes) if rule.attributes else "{}"
        filter_str = str(rule.filter) if rule.filter else "{}"
        return f"""You are a symbolic rule optimizer.

The following rule has been applied {len(applications)} times with an average score of {statistics.mean(scores):.2f}.

Filter: {filter_str}
Attributes: {attributes_str}

Here are some example scores: {scores[:5]}

Please suggest improvements to this rule (e.g., modify attributes, adjust filter constraints, or recommend deprecation if not useful). Return only the proposed change."""
