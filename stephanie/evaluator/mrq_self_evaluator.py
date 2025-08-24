# stephanie/evaluator/mrq_self_evaluator.py

from stephanie.data.score_bundle import ScoreBundle
from stephanie.evaluator.base import BaseEvaluator
from stephanie.scoring.scorable_factory import ScorableFactory


class MRQSelfEvaluator(BaseEvaluator):
    """
    Temporary wrapper: MRQ evaluator delegating to SICQLScorer.
    Keeps the same interface so existing code won't break.
    """

    def __init__(self, cfg, memory, logger, device="cpu"):
        from stephanie.scoring.sicql_scorer import SICQLScorer
        self.cfg = cfg
        self.device = device
        self.memory = memory
        self.logger = logger

        # Just use SICQL underneath for now
        self.scorer = SICQLScorer(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", ["value"])

    def judge(self, prompt, output_a, output_b, context=None):
        scorable_a = ScorableFactory.from_text(output_a, target_type="response")
        scorable_b = ScorableFactory.from_text(output_b, target_type="response")

        bundle_a: ScoreBundle = self.scorer.score(context, scorable_a, self.dimensions)
        bundle_b: ScoreBundle = self.scorer.score(context, scorable_b, self.dimensions)

        # Take first dimension as decision driver
        dim = self.dimensions[0]
        value_a = bundle_a[dim].score
        value_b = bundle_b[dim].score

        preferred = output_a if value_a >= value_b else output_b
        scores = {"value_a": value_a, "value_b": value_b}
        return preferred, scores

    def score_single(self, prompt: str, output: str, context: dict) -> float:
        scorable = ScorableFactory.from_text(output, target_type="response")
        bundle = self.scorer.score(context, scorable, self.dimensions)
        return bundle[self.dimensions[0]].score

    def train_from_database(self, goal: str, cfg: dict):
        """Train the model using data from a database."""
        pass

    def train_from_context(self, context: dict, cfg: dict):
        """Train the model using the current context."""    
        pass
