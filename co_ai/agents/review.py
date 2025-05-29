# co_ai/agents/review.py
from co_ai.agents.base import BaseAgent
from co_ai.constants import HYPOTHESES, REVIEW, TEXT
from co_ai.scoring.review import ReviewScore


class ReviewAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)
        reviews = []
        review_scorer = ReviewScore(self.cfg, memory=self.memory, logger=self.logger)
        for hyp in hypotheses:
            hyp_text = hyp.get(TEXT)
            prompt = self.prompt_loader.load_prompt(self.cfg, {**context, **{HYPOTHESES:hyp_text}})
            review = self.call_llm(prompt, context)
            hyp_id = self.get_hypothesis_id(hyp)
            self.memory.hypotheses.update_review(hyp_id, review)
            hyp[REVIEW] =review
            score = review_scorer.get_score(hyp, context)
            reviews.append(review)
            self.logger.log(
                "ReviewScoreComputed", {"hypothesis_id": hyp_id, "score": score, "review": review}
            )

        context[self.output_key] = reviews
        return context
