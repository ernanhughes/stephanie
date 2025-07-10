from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.scoring_manager import ScoringManager


class ScoringMixin:
    """
    A generic scoring mixin that supports dynamic, stage-aware evaluation using ScoreEvaluator.

    Supports any configured scoring stage (e.g., review, reasoning, reflection).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scorers = {}  # Caches ScoreEvaluator instances per stage

    def get_scorer(self, scoring_profile: str) -> ScoringManager:
        """
        Lazily loads and returns a ScoreEvaluator for the given stage.
        Config path is read from e.g., cfg['review_score_config'].
        """
        if scoring_profile not in self._scorers:
            config_key = f"{scoring_profile}_score_config"
            config_path = self.cfg.get(config_key, f"config/scoring/{scoring_profile}.yaml")
            self._scorers[scoring_profile] = ScoringManager.from_file(
                filepath=config_path,
                prompt_loader=self.prompt_loader,
                cfg=self.cfg,
                logger=self.logger,
                memory=self.memory
            )
        return self._scorers[scoring_profile]

    def get_dimensions(self, scoing_profile: str) -> ScoringManager:
        """
        Lazily loads and returns a ScoreEvaluator for the given stage.
        Config path is read from e.g., cfg['review_score_config'].
        """
        if scoing_profile not in self._scorers:
            config_key = f"{scoing_profile}_score_config"
            config_path = self.cfg.get(config_key, f"config/scoring/{scoing_profile}.yaml")
            self._scorers[scoing_profile] = ScoringManager.from_file(
                filepath=config_path,
                prompt_loader=self.prompt_loader,
                cfg=self.cfg,
                logger=self.logger,
                memory=self.memory
            )
        return self._scorers[scoing_profile]
    
    def get_dimensions(self, scoring_profile: str) -> list:
        """
        Get the list of dimensions for a given scoring stage.
        
        Args:
            stage (str): The scoring stage (e.g., "review", "reasoning", "reflection").
        
        Returns:
            list: List of dimension names for the specified stage.
        """
        scorer = self.get_scorer(scoring_profile)
        return scorer.get_dimensions()

    def score_item(
        self,
        scorable: Scorable,
        context: dict,
        metrics: str = "review",
        scorer: BaseScorer = None,
    ) -> ScoreBundle:
        """
        Score a hypothesis for a given evaluation stage.

        Args:
            hypothesis:
            hyp (dict): Hypothesis object with a "text" key.
            context (dict): Pipeline context, must include 'goal'.
            metrics (str): Evaluation metrics (e.g., "review", "reasoning", "reflection").
            evaluator (callable): Optional evaluator override (e.g., a parser function).

        Returns:
            dict: {
                "id": hypothesis_id,
                "score": float,
                "scores": {dimension_name: {score, rationale, weight}, ...},
                "metrics": metrics
            }
        """
        default_scorer = self.get_scorer(metrics)

        if scorer:
            dimensions = default_scorer.get_dimensions()
            result = scorer.score(context.get("goal"), scorable, dimensions)
            self.logger.log("HypothesisScored", result.to_dict())
        else:
            result = default_scorer.evaluate(
                scorable=scorable,
                context=context,
                llm_fn=self.call_llm
            )
            self.logger.log("HypothesisScored", result.to_dict())

        return result