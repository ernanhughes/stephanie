from co_ai.analysis.score_evaluator import ScoreEvaluator
from co_ai.scoring.base_evaluator import BaseEvaluator


class ScoringMixin:
    """
    A generic scoring mixin that supports dynamic, stage-aware evaluation using ScoreEvaluator.

    Supports any configured scoring stage (e.g., review, reasoning, reflection).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scorers = {}  # Caches ScoreEvaluator instances per stage

    def get_scorer(self, stage: str) -> ScoreEvaluator:
        """
        Lazily loads and returns a ScoreEvaluator for the given stage.
        Config path is read from e.g., cfg['review_score_config'].
        """
        if stage not in self._scorers:
            config_key = f"{stage}_score_config"
            config_path = self.cfg.get(config_key, f"config/scoring/{stage}.yaml")
            self._scorers[stage] = ScoreEvaluator.from_file(
                filepath=config_path,
                prompt_loader=self.prompt_loader,
                cfg=self.cfg,
                logger=self.logger,
                memory=self.memory
            )
        return self._scorers[stage]

    def score_hypothesis(
        self,
        hypothesis: dict,
        context: dict,
        metrics: str = "review",
        evaluator: BaseEvaluator = None,
    ) -> dict:
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
        if evaluator:
            result = evaluator.evaluate(hypothesis, context)
            final_score = result["score"]
            dimension_scores = result.get("dimensions", {
                metrics: {
                    "score": final_score,
                    "weight": 1.0,
                    "rationale": result.get("rationale", "")
                }
            })
        else:
            scorer = self.get_scorer(metrics)
            dimension_scores = scorer.evaluate(
                hypothesis=hypothesis,
                context=context,
                llm_fn=self.call_llm
            )

        weighted_total = sum(
            s["score"] * s.get("weight", 1.0)
            for s in dimension_scores.values()
        )
        weight_sum = sum(s.get("weight", 1.0) for s in dimension_scores.values())
        final_score = round(weighted_total / weight_sum, 2) if weight_sum > 0 else 0.0

        self.logger.log("HypothesisScoreComputed", {
            "score": final_score,
            "dimension_scores": dimension_scores,
            "hypothesis": hypothesis,
            "metrics": metrics
        })

        return {
            "id": hypothesis.get("id"),
            "score": final_score,
            "scores": dimension_scores,
            "metrics": metrics
        }
