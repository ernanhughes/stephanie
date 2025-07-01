from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin


class ReviewAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)
        reviews = []

        for hyp in hypotheses:
            # Score and update review
            score = self.score_hypothesis(hyp, context, metrics="review")
            self.logger.log(
                "ReviewScoreComputed",
                score,
            )
            reviews.append(score)

        context[self.output_key] = reviews
        return context