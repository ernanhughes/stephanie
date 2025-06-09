from co_ai.agents.base import BaseAgent
from co_ai.agents.mixins.scoring_mixin import ScoringMixin


class ReflectionAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)

        reflections = []
        for hyp in hypotheses:
            score = self.score_hypothesis(hyp, context, metrics="reflection")
            self.logger.log(
                "ReflectionScoreComputed",
                score,
            )
            reflections.append(score)

        context[self.output_key] = reflections
        return context