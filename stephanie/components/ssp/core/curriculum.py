from __future__ import annotations

from collections import deque
from typing import Deque


class QMaxCurriculum:
    """
    Simple Q* (Q-Max) style curriculum control:
      - Track recent success rate in a fixed window
      - Raise difficulty when agent is competent, lower when struggling
    """

    def __init__(
        self,
        initial: float = 0.30,
        step: float = 0.05,
        maximum: float = 0.95,
        window: int = 100,
        min_success_rate: float = 0.60,
    ):
        self.difficulty = float(initial)
        self._step = float(step)
        self._max = float(maximum)
        self._min = float(initial)
        self._window = int(window)
        self._min_success = float(min_success_rate)
        self._hist: Deque[int] = deque(maxlen=self._window)

    @property
    def success_rate(self) -> float:
        return (sum(self._hist) / len(self._hist)) if self._hist else 0.0

    def update(self, success: bool) -> float:
        self._hist.append(1 if success else 0)
        if self.success_rate >= self._min_success:
            self.difficulty = min(self._max, self.difficulty + self._step)
        else:
            self.difficulty = max(self._min, self.difficulty - self._step)
        return self.difficulty

    def observe(self, success: bool) -> float:
        self.history.append(1 if success else 0)
        if not self.history:
            return self.difficulty
        sr = sum(self.history) / len(self.history)
        if sr >= self.min_sr:
            self.difficulty = min(self.max_val, self.difficulty + self.step)
        elif sr < self.min_sr * 0.7:
            self.difficulty = max(self.min_val, self.difficulty - self.step * 2)
        return self.difficulty

    def get(self) -> float:
        return self.difficulty
