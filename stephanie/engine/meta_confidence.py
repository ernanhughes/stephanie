# stephanie/engine/meta_confidence.py

from collections import defaultdict


class MetaConfidenceTracker:
    """
    Tracks model trustworthiness and confidence across goals and dimensions.
    Based on validation agreement, prediction margins, and stability over time.
    """

    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        self.history = defaultdict(list)  # (goal, dimension) → list of validation results

        # Configurable thresholds
        self.retrain_threshold = getattr(cfg, "retrain_threshold", 0.65)
        self.fallback_threshold = getattr(cfg, "fallback_threshold", 0.50)

    def update(
        self,
        goal: str,
        dimension: str,
        validation_result: dict
    ):
        """
        Accepts results from SelfValidationEngine and updates internal trust score.
        """
        key = (goal, dimension)
        self.history[key].append(validation_result)

        if self.logger:
            self.logger.info("MetaConfidenceUpdated", extra={
                "goal": goal,
                "dimension": dimension,
                "agreement": validation_result.get("agreement"),
                "validated": validation_result.get("validated")
            })

        # Optional: persist to memory
        if self.memory:
            self.memory.save("meta_confidence", {
                "goal": goal,
                "dimension": dimension,
                "agreement": validation_result.get("agreement")
            })

    def get_confidence(self, goal: str, dimension: str) -> float:
        """
        Returns average agreement score over recent history for a goal/dimension.
        """
        key = (goal, dimension)
        recent = self.history[key][-10:]  # last 10 cycles
        if not recent:
            return 1.0  # Assume trust until proven otherwise
        scores = [r["agreement"] for r in recent if "agreement" in r]
        return round(sum(scores) / len(scores), 3) if scores else 1.0

    def should_fallback(self, goal: str, dimension: str) -> bool:
        """
        Should we fallback to the LLM instead of trusting the model?
        """
        return self.get_confidence(goal, dimension) < self.fallback_threshold

    def should_retrain(self, goal: str, dimension: str) -> bool:
        """
        Should we trigger a retraining event for this goal/dimension?
        """
        confidence = self.get_confidence(goal, dimension)
        return 0 < confidence < self.retrain_threshold
