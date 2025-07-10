from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType


class ReviewAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)
        reviews = []

        for hyp in hypotheses:
            # Score and update review
            scorable = Scorable(id=hyp.get("id", ""), text=hyp.get("text", ""), target_type=TargetType.HYPOTHESIS)
            score = self.score_item(scorable, context, metrics="review")
            self.logger.log(
                "ReviewScoreComputed",
                score,
            )
            reviews.append(score.to_dict())

        context[self.output_key] = reviews
        return context