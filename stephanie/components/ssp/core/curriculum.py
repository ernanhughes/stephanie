# stephanie/components/ssp/core/curriculum.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple
from collections import deque
import math
import statistics as stats

@dataclass
class QMaxCurriculum:
    """
    Simple, bounded-window curriculum:
    - Track recent episode returns and success_rate
    - Map to difficulty via target-success band
    - Exposes a small state dict for telemetry
    """
    window: int = 200
    target_success: float = 0.65
    min_diff: float = 0.1
    max_diff: float = 1.0
    _returns: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    _successes: Deque[int] = field(default_factory=lambda: deque(maxlen=200))
    _d: float = 0.3  # current difficulty (0..1)

    def update(self, episode_return: float, success: bool) -> None:
        self._returns.append(float(episode_return))
        self._successes.append(1 if success else 0)
        self._recalc()

    # Backwards-compat shim if callers use 'observe'
    def observe(self, episode_return: float, success: bool) -> None:
        self.update(episode_return, success)

    def _recalc(self) -> None:
        if not self._successes:
            return
        sr = sum(self._successes) / len(self._successes)
        # Move difficulty toward keeping sr near target
        err = (self.target_success - sr)
        step = 0.15 * err  # proportional control
        self._d = float(min(self.max_diff, max(self.min_diff, self._d + step)))

    def difficulty(self) -> float:
        return self._d

    def snapshot(self) -> Dict:
        if self._returns:
            mean = stats.fmean(self._returns)
            sd = stats.pstdev(self._returns) if len(self._returns) > 1 else 0.0
        else:
            mean, sd = 0.0, 0.0
        sr = (sum(self._successes) / len(self._successes)) if self._successes else 0.0
        return {
            "difficulty": self._d,
            "success_rate": sr,
            "return_mean": mean,
            "return_sd": sd,
            "n": len(self._returns),
        }
