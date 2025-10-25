# stephanie/scoring/meta_review_scorer.py
from __future__ import annotations

from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.scorer.llm_scorer import LLMScorer
from stephanie.scoring.scorer.mrq_scorer import \
    MRQScorer  # formerly MRQEvaluator


class MetaReviewScorer(BaseScorer):
    """
    Combines MR.Q-based scoring with LLM fallback.

    Tries MRQ first. If it's missing dimensions or returns low-confidence scores, falls back to LLM.
    """

    def __init__(self, cfg, memory, logger, container, fallback_to_llm=True):
        super().__init__(cfg, memory, container, logger)
        self.memory = memory
        self.logger = logger
        self.cfg = cfg or {}
        self.use_llm_fallback = fallback_to_llm

        self.mrq_scorer = MRQScorer(cfg, memory, container, logger)
        self.llm_scorer = LLMScorer(cfg, memory, container, logger)

    def _score_core(self, goal, hypothesis, dimensions):
        mrq_scores = self.mrq_scorer.score(goal, hypothesis, dimensions)

        if self._needs_llm_fallback(mrq_scores, dimensions):
            self.logger.log(
                "MetaReviewFallbackTriggered",
                {"reason": "Missing or low-confidence MRQ scores", "fallback": "llm"},
            )
            llm_scores = self.llm_scorer.score(goal, hypothesis, dimensions)
            return self._combine_scores(mrq_scores, llm_scores)
        else:
            return mrq_scores

    def _needs_llm_fallback(self, scores, dimensions):
        """
        Trigger fallback if:
        - Any dimension is missing in MRQ
        - Any score is None or clearly untrained (e.g., 0.0)
        """
        for dim in dimensions:
            if dim not in scores:
                return True
            if abs(scores[dim]["score"] - 0.0) < 1e-8:
                return True
        return False

    def _combine_scores(self, mrq_scores, llm_scores):
        """
        Merge: prefer MRQ if valid, otherwise use LLM.
        """
        combined = {}
        all_dims = set(mrq_scores.keys()) | set(llm_scores.keys())
        for dim in all_dims:
            if dim in mrq_scores and mrq_scores[dim]["score"] > 0.0:
                combined[dim] = mrq_scores[dim]
            else:
                combined[dim] = llm_scores.get(
                    dim,
                    {"score": 0.0, "rationale": "No fallback available", "weight": 1.0},
                )
        return combined
