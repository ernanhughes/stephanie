# stephanie/services/meta_confidence_service.py
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict

from stephanie.services.service_protocol import Service


class MetaConfidenceService(Service):
    """
    Service for tracking model trustworthiness and confidence across goals and dimensions.

    Tracks:
    - Validation agreement
    - Prediction margins
    - Stability over time

    Provides:
    - Confidence scores
    - Fallback triggers
    - Retraining triggers
    """

    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        self.history = defaultdict(list)  # (goal, dimension) → list of validation results

        # Configurable thresholds
        self.retrain_threshold = getattr(cfg, "retrain_threshold", 0.65)
        self.fallback_threshold = getattr(cfg, "fallback_threshold", 0.50)

        self._initialized = False

    # === Service Protocol ===
    def initialize(self, **kwargs) -> None:
        """Allocate resources (no heavy resources here, but mark initialized)."""
        self._initialized = True
        if self.logger:
            self.logger.log("MetaConfidenceInit", {"status": "initialized"})

    def health_check(self) -> Dict[str, Any]:
        """Return service health + recent metrics."""
        status = "healthy" if self._initialized else "unhealthy"
        recent_counts = {str(k): len(v) for k, v in self.history.items()}

        return {
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "tracked_dimensions": len(self.history),
                "recent_updates": sum(recent_counts.values()),
            },
            "dependencies": {},
        }

    def shutdown(self) -> None:
        """Gracefully shut down service and clear history."""
        self.history.clear()
        self._initialized = False
        if self.logger:
            self.logger.log("MetaConfidenceShutdown", {})

    @property
    def name(self) -> str:
        return "confidence-tracker-v1"

    # === Domain Logic ===
    def update(self, goal: str, dimension: str, validation_result: dict):
        """
        Accepts results from SelfValidationEngine and updates internal trust score.
        """
        key = (goal, dimension)
        self.history[key].append(validation_result)

        if self.logger:
            self.logger.log(
                "MetaConfidenceUpdated",
                extra={
                    "goal": goal,
                    "dimension": dimension,
                    "agreement": validation_result.get("agreement"),
                    "validated": validation_result.get("validated"),
                },
            )

        # Optional: persist to memory
        if self.memory:
            self.memory.save(
                "meta_confidence",
                {
                    "goal": goal,
                    "dimension": dimension,
                    "agreement": validation_result.get("agreement"),
                },
            )

    def get_confidence(self, goal: str, dimension: str) -> float:
        """Returns average agreement score over recent history for a goal/dimension."""
        key = (goal, dimension)
        recent = self.history[key][-10:]  # last 10 cycles
        if not recent:
            return 1.0  # Assume trust until proven otherwise
        scores = [r["agreement"] for r in recent if "agreement" in r]
        return round(sum(scores) / len(scores), 3) if scores else 1.0

    def should_fallback(self, goal: str, dimension: str) -> bool:
        """Should we fallback to the LLM instead of trusting the model?"""
        return self.get_confidence(goal, dimension) < self.fallback_threshold

    def should_retrain(self, goal: str, dimension: str) -> bool:
        """Should we trigger a retraining event for this goal/dimension?"""
        confidence = self.get_confidence(goal, dimension)
        return 0 < confidence < self.retrain_threshold
