# stephanie/agents/reflection.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_scorables(context)

        reflections = []
        for hyp in hypotheses:
            scorable = ScorableFactory.from_dict(hyp, TargetType.HYPOTHESIS)
            score = self._score(scorable=scorable, context=context)
            self.logger.log(
                "ReflectionScoreComputed",
                score,
            )
            reflections.append(score)

        context[self.output_key] = reflections
        return context
