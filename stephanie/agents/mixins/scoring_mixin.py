# stephanie/agents/mixins/scoring_mixin.py

from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.scoring_manager import ScoringManager


class ScoringMixin:
    """
    A generic scoring mixin that supports dynamic, stage-aware evaluation using ScoreEvaluator
    or directly with MRQScorer / LLMScorer.

    Supports any configured scoring stage (e.g., review, reasoning, reflection).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scorers = {}  # Caches ScoringManager instances per profile

    def get_scorer(self, scoring_profile: str) -> ScoringManager:
        """
        Lazily loads and returns a ScoringManager for the given profile (e.g. review, reasoning).
        """
        if scoring_profile not in self._scorers:
            config_key = f"{scoring_profile}_score_config"
            config_path = self.cfg.get(
                config_key, f"config/scoring/{scoring_profile}.yaml"
            )
            self._scorers[scoring_profile] = ScoringManager.from_file(
                filepath=config_path,
                prompt_loader=self.prompt_loader,
                cfg=self.cfg,
                logger=self.logger,
                memory=self.memory,
            )
        return self._scorers[scoring_profile]

    def get_dimensions(self, scoring_profile: str) -> list[str]:
        """
        Returns the list of scoring dimensions from the ScoringManager config.
        """
        return self.get_scorer(scoring_profile).get_dimensions()

    def score_item(
        self,
        scorable: Scorable,
        context: dict,
        metrics: str = "review",
        scorer: BaseScorer = None,
    ) -> ScoreBundle:
        """
        Score a scorable item for a given scoring stage (e.g., review, reasoning).
        Uses either a passed-in scorer (like MRQScorer) or a config-based ScoringManager.

        Returns:
            ScoreBundle
        """
        if scorer:
            bundle = scorer.score(context.get("goal"), scorable, metrics)
        else:
            # Default to full ScoringManager logic (LLM-driven)
            scoring_manager = self.get_scorer(metrics)
            bundle = scoring_manager.evaluate(
                scorable=scorable, context=context, llm_fn=self.call_llm
            )

        self.logger.log("HypothesisScored", bundle.to_dict())
        return bundle
