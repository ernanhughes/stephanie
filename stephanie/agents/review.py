# stephanie/agents/review.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import ScorableFactory, ScorableType


class ReviewAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = context.get(self.input_key, [])
        reviews = []

        for hyp in hypotheses:
            # Score and update review
            scorable = ScorableFactory.from_dict(hyp, ScorableType.HYPOTHESIS)
            score = self._score(scorable=scorable, context=context)
            self.logger.log(
                "ReviewScoreComputed",
                score,
            )
            reviews.append(score.to_dict())

        context[self.output_key] = reviews
        return context
