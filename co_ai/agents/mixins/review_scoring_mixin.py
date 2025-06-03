from co_ai.scoring.review import ReviewScore
from co_ai.constants import HYPOTHESES, REVIEW


class ReviewScoringMixin:
    """
    A mixin that provides review scoring functionality to any agent.
    Can be used in ReviewAgent, MetaReviewAgent, or any composite evaluator agent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.review_scorer = None  # Will be initialized on first use

    def get_review_scorer(self):
        """
        Lazily initialize the ReviewScore instance.
        """
        if not self.review_scorer:
            self.review_scorer = ReviewScore(
                self.cfg, memory=self.memory, logger=self.logger
            )
        return self.review_scorer

    def score_hypothesis_with_review(self, hyp: dict, context: dict) -> dict:
        """
        Score a hypothesis using its review text.

        Args:
            hyp (dict): Hypothesis dictionary containing "text" and optionally "review".
            context (dict): Execution context for prompt generation or metadata.

        Returns:
            float: The computed score.
        """
        hyp_text = hyp.get("text")
        hyp_id = self.get_hypothesis_id(hyp)

        # If no review exists yet, generate one
        if not hyp.get(REVIEW):
            prompt = self.prompt_loader.load_prompt(
                self.cfg, {**context, **{HYPOTHESES: hyp_text}}
            )
            review = self.call_llm(prompt, context)
            hyp[REVIEW] = review
            self.memory.hypotheses.update_review(hyp_id, review)
        else:
            review = hyp[REVIEW]

        # Score the hypothesis based on the review
        scorer = self.get_review_scorer()
        score = scorer.get_score(hyp, context)
        # Log the scoring event
        self.logger.log("ReviewScoreComputed", {"hypothesis_id": hyp_id, "score": score, "review": review})
        return {"id": hyp_id, "score": score, REVIEW: review, "scores": scorer.scores}
