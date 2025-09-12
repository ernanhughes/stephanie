# stephanie/services/state_tracker_service.py
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict

from stephanie.services.service_protocol import Service


class StateTrackerService(Service):
    """
    Service for tracking goal/dimension state over time.

    Tracks:
    - Scoring
    - Validation
    - Training events
    - Goal metadata
    - Frozen/active status

    Provides:
    - Event history
    - Safe learning coordination
    - Decision gating
    """

    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.state = defaultdict(dict)
        self._initialized = False

    # === Service Protocol ===
    def initialize(self, **kwargs) -> None:
        """Initialize the state tracker service."""
        self._initialized = True
        if self.logger:
            self.logger.log("StateTrackerInit", {"status": "initialized"})

    def health_check(self) -> Dict[str, Any]:
        """Return service health + tracked goals/dimensions summary."""
        status = "healthy" if self._initialized else "unhealthy"

        return {
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "tracked_goals": len({g for (g, _) in self.state.keys()}),
                "tracked_dimensions": len(self.state),
            },
            "dependencies": {},
        }

    def shutdown(self) -> None:
        """Clear all tracked state."""
        self.state.clear()
        self._initialized = False
        if self.logger:
            self.logger.log("StateTrackerShutdown", {})

    @property
    def name(self) -> str:
        return "state-tracker-v1"

    # === Domain Logic ===
    def update_event(self, goal: str, dimension: str, event_type: str):
        """Update state based on event type (scored, validated, trained, frozen, active)."""
        key = (goal, dimension)
        now = time.time()

        if event_type == "scored":
            self.state[key]["last_scored_at"] = now
        elif event_type == "validated":
            self.state[key]["last_validated_at"] = now
        elif event_type == "trained":
            self.state[key]["last_trained_at"] = now
            self.state[key]["retrain_count"] = (
                self.state[key].get("retrain_count", 0) + 1
            )
        elif event_type == "frozen":
            self.state[key]["status"] = "frozen"
        elif event_type == "active":
            self.state[key]["status"] = "active"

        if self.logger:
            self.logger.log(
                "GoalStateUpdated",
                extra={
                    "goal": goal,
                    "dimension": dimension,
                    "event": event_type,
                    "timestamp": now,
                },
            )

        if self.memory:
            self.memory.save(
                "state_tracker",
                {"goal": goal, "dimension": dimension, "state": self.state[key]},
            )

    def get_state(self, goal: str, dimension: str) -> dict:
        """Return current state for (goal, dimension)."""
        return self.state.get((goal, dimension), {})

    def is_new_goal(self, goal: str) -> bool:
        """Check if goal has been seen before."""
        for (g, _), _ in self.state.items():
            if g == goal:
                return False
        return True

    def mark_goal_metadata(self, goal: str, metadata: dict):
        """Attach metadata to a goal."""
        self.state[(goal, None)]["metadata"] = metadata
        if self.memory:
            self.memory.save("goal_metadata", {"goal": goal, "metadata": metadata})

    def __repr__(self):
        summary = []
        for (goal, dimension), state in self.state.items():
            dim_str = dimension if dimension is not None else "ALL"
            status = state.get("status", "unknown")
            scored = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(state.get("last_scored_at", 0))
            )
            trained = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(state.get("last_trained_at", 0))
            )
            retrain_count = state.get("retrain_count", 0)
            summary.append(
                f"[{goal} | {dim_str}] status={status}, last_scored={scored}, "
                f"last_trained={trained}, retrains={retrain_count}"
            )
        return f"<StateTrackerService: {len(summary)} tracked goals>\n" + "\n".join(summary)
