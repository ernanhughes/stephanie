from co_ai.agents.base import BaseAgent
from co_ai.agents.mixins.reflection_scoring_mixin import ReflectionScoringMixin


class ReflectionAgent(ReflectionScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)

        reflections = []
        for hyp in hypotheses:
            score = self.score_hypothesis_with_reflection(hyp, context)
            reflections.append(score)

        context[self.output_key] = reflections
        return context