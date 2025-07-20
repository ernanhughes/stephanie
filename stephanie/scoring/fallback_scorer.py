# File: stephanie/scoring/fallback_scorer.py

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

from stephanie.constants import GOAL
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle


@dataclass
class ScoreAttempt:
    scorer_name: str
    success: bool
    error: Optional[str] = None
    score_bundle: Optional[ScoreBundle] = None
    timestamp: datetime = datetime.utcnow()


class FallbackScorer(BaseScorer):
    """
    A composite scorer that tries multiple scorers in order of preference.
    Falls back when a scorer fails due to missing model, data, or timeout.
    """

    def __init__(
        self,
        scorers: List[BaseScorer],
        fallback_order: Optional[List[str]] = None,
        default_fallback: str = "llm",
        logger = None,
    ): 
        """
        Args:
            scorers: List of scorers to try
            fallback_order: Order of scorer preference (e.g. ["svm", "mrq", "llm"])
            default_fallback: Final fallback if all scorers fail
            logger: Optional logger
        """
        super().__init__(cfg={}, memory=None, logger=logger)
        self.scorers = {scorer.name: scorer for scorer in scorers}
        self.fallback_order = fallback_order or list(self.scorers.keys())
        self.default_fallback = default_fallback

    def score(self, context: dict, scorable: Scorable, dimensions: List[str] = None) -> ScoreBundle:
        """
        Try scorers in order. Returns first successful score bundle.
        If all fail, returns neutral score with fallback reason.
        """
        goal = context.get(GOAL, {})
        scorer_used = None
        attempts = []

        for scorer_name in self.fallback_order:
            scorer = self.scorers.get(scorer_name)

            if not scorer:
                self.logger.log(
                    "ScorerNotRegistered",
                    {"scorer_name": scorer_name, "available_scorers": list(self.scorers.keys())}
                )
                continue

            try:
                self.logger.log("TryingScorer", {"scorer": scorer_name, "target": scorable.id})
                score_bundle = scorer.score(context, scorable, dimensions=dimensions)

                if score_bundle.is_valid():
                    scorer_used = scorer_name
                    attempts.append(ScoreAttempt(
                        scorer_name=scorer_name,
                        success=True,
                        score_bundle=score_bundle
                    ))
                    self.logger.log(
                        "ScoreSuccess",
                        {"scorer": scorer_name, "target": scorable.id, "scores": score_bundle.to_dict()}
                    )
                    break
                else:
                    attempts.append(ScoreAttempt(
                        scorer_name=scorer_name,
                        success=False,
                        error="Invalid score bundle"
                    ))
                    self.logger.log(
                        "ScoreInvalid",
                        {"scorer": scorer_name, "target": scorable.id}
                    )

            except Exception as e:
                attempts.append(ScoreAttempt(
                    scorer_name=scorer_name,
                    success=False,
                    error=str(e)
                ))
                self.logger.log(
                    "ScoreFailed",
                    {"scorer": scorer_name, "target": scorable.id, "error": str(e)}
                )
                continue

        if scorer_used:
            return score_bundle

        # Final fallback: use default scorer
        default_scorer = self.scorers.get(self.default_fallback)
        if default_scorer:
            self.logger.log(
                "FinalFallbackUsed",
                {"scorer": self.default_fallback, "target": scorable.id}
            )
            return default_scorer.score(context, scorable, dimensions=dimensions)

        # If even fallback fails, return neutral score
        self.logger.log(
            "AllScorersFailed",
            {"target": scorable.id, "attempts": [a.scorer_name for a in attempts]}
        )
        return ScoreBundle.from_dict({
            dim: 0.5 for dim in dimensions or ["usefulness", "novelty", "alignment"]
        })

    def load_models(self):
        """Load models for all scorers."""
        for scorer in self.scorers.values():
            try:
                scorer.load_models()
            except Exception as e:
                self.logger.log("ModelLoadFailed", {"scorer": scorer.name, "error": str(e)})

    def train(self, samples_by_dim: Dict[str, list]):
        """Train all scorers that support training."""
        for scorer in self.scorers.values():
            try:
                scorer.train(samples_by_dim)
            except Exception as e:
                self.logger.log("TrainingFailed", {"scorer": scorer.name, "error": str(e)})

    def save_models(self):
        """Save models for all scorers."""
        for scorer in self.scorers.values():
            try:
                scorer.save_models()
            except Exception as e:
                self.logger.log("ModelSaveFailed", {"scorer": scorer.name, "error": str(e)})
