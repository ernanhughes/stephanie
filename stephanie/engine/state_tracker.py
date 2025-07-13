# stephanie/engine/state_tracker.py

import time
from collections import defaultdict


class StateTracker:
    """
    Tracks goal/dimension state over time — scoring, validation, training, metadata.
    Used to coordinate safe learning, logging, and decision gating.
    """

    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.state = defaultdict(dict)

    def update_event(self, goal: str, dimension: str, event_type: str):
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
            self.logger.info(
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
        return self.state.get((goal, dimension), {})

    def is_new_goal(self, goal: str) -> bool:
        for (g, _), _ in self.state.items():
            if g == goal:
                return False
        return True

    def mark_goal_metadata(self, goal: str, metadata: dict):
        self.state[(goal, None)]["metadata"] = metadata
        if self.memory:
            self.memory.save("goal_metadata", {"goal": goal, "metadata": metadata})
