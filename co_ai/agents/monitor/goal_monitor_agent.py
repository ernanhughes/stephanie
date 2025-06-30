# co_ai/agents/monitor/goal_monitor_agent.py
from co_ai.registry.registry import get


class GoalMonitorAgent:
    """
    Oversees the learning and performance status of all goals.
    Integrates validation, confidence, cycles, and state to make safe decisions.
    """

    def __init__(
        self,
        cfg,
        memory,
        logger
    ):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        self.state_tracker = get("state_tracker")
        self.confidence_tracker = get("confidence_tracker")
        self.cycle_watcher = get("cycle_watcher")
        self.training_controller = get("training_controller")

        self.freeze_threshold = getattr(cfg, "goal_freeze_threshold", 0.45)

    def monitor_goal(self, goal: str, dimension: str, training_pairs: list[dict]):
        """
        Main entry point: validates performance, updates trackers, controls retraining,
        and potentially freezes a goal if it proves unstable.
        """

        # Run training controller cycle
        self.training_controller.maybe_train(goal, dimension, training_pairs)

        # Get state
        confidence = self.confidence_tracker.get_confidence(goal, dimension)
        cycle_status = self.cycle_watcher.status(goal, dimension)
        state = self.state_tracker.get_state(goal, dimension)

        # Detect frozen or degraded goals
        if confidence < self.freeze_threshold and cycle_status == "oscillating":
            self.logger.warning("Freezing goal due to instability", extra={
                "goal": goal,
                "dimension": dimension,
                "confidence": confidence,
                "status": cycle_status
            })
            self.state_tracker.update_event(goal, dimension, event_type="frozen")
            return "frozen"

        # Otherwise keep active
        self.state_tracker.update_event(goal, dimension, event_type="active")
        return {
            "goal": goal,
            "dimension": dimension,
            "status": "active",
            "confidence": confidence,
            "cycle": cycle_status,
            "retrain_count": state.get("retrain_count", 0)
        }
