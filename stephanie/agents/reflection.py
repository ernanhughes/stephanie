# stephanie/agents/reflection.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType


class ReflectionAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)

        reflections = []
        for hyp in hypotheses:
            scorable = ScorableFactory.from_dict(hyp, TargetType.HYPOTHESIS)
            score = self.score_item(scorable, context, metrics="reflection")
            self.logger.log(
                "ReflectionScoreComputed",
                score,
            )
            reflections.append(score)

        context[self.output_key] = reflections
        return context
