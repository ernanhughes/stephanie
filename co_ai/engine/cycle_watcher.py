# co_ai/engine/cycle_watcher.py

from collections import defaultdict, deque
from statistics import mean


class CycleWatcher:
    """
    Monitors goal/dimension performance over time to detect stagnation, oscillation, or unstable loops.
    """

    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        self.history = defaultdict(lambda: deque(maxlen=20))  # (goal, dimension) → agreement scores

        # Tuning thresholds
        self.plateau_delta = getattr(cfg, "cycle_plateau_delta", 0.01)
        self.plateau_window = getattr(cfg, "cycle_plateau_window", 5)
        self.oscillation_window = getattr(cfg, "cycle_oscillation_window", 4)

    def record_agreement(self, goal: str, dimension: str, agreement: float):
        key = (goal, dimension)
        self.history[key].append(agreement)

        if self.logger:
            self.logger.info("CycleAgreementTracked", extra={
                "goal": goal,
                "dimension": dimension,
                "agreement": agreement
            })

    def is_stuck(self, goal: str, dimension: str) -> bool:
        """
        Detects a plateau — no net improvement over last N rounds.
        """
        key = (goal, dimension)
        scores = list(self.history[key])[-self.plateau_window:]
        if len(scores) < self.plateau_window:
            return False
        return max(scores) - min(scores) < self.plateau_delta

    def is_oscillating(self, goal: str, dimension: str) -> bool:
        """
        Detects frequent back-and-forth confidence shifts.
        """
        key = (goal, dimension)
        scores = list(self.history[key])[-self.oscillation_window:]
        if len(scores) < self.oscillation_window:
            return False
        # Count sign changes in delta
        deltas = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        sign_changes = sum(
            1 for i in range(len(deltas) - 1)
            if deltas[i] * deltas[i + 1] < 0
        )
        return sign_changes >= 2  # e.g. up-down-up or down-up-down

    def status(self, goal: str, dimension: str) -> str:
        """
        Returns current state for a goal+dimension: 'stable', 'stuck', or 'oscillating'
        """
        if self.is_stuck(goal, dimension):
            return "stuck"
        if self.is_oscillating(goal, dimension):
            return "oscillating"
        return "stable"
