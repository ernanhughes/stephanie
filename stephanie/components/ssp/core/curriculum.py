# stephanie/components/ssp/core/curriculum.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple, Union
from collections import deque
import statistics as stats

Number = Union[int, float]

@dataclass
class QMaxCurriculum:
    """
    Simple, bounded-window curriculum with backward compatibility.

    - Tracks recent episode returns and success flags
    - Proportional control keeps success rate near a target by adjusting difficulty
    - Compatible with callers that:
        * call update(reward: float, success: bool)
        * call update(success: bool)  # legacy pattern
    - Exposes .success_rate and .difficulty (property alias .value) for consumers
    """
    window: int = 200
    target_success: float = 0.65
    min_diff: float = 0.1
    max_diff: float = 1.0

    _returns: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    _successes: Deque[int] = field(default_factory=lambda: deque(maxlen=200))
    _d: float = 0.3  # current difficulty in [min_diff, max_diff]

    # ----------------------------- public API -----------------------------

    def update(self,
               episode_return: Optional[Union[Number, bool]] = None,
               success: Optional[bool] = None) -> float:
        """
        Update curriculum state and return the current difficulty.

        Accepts either:
          - update(reward: float, success: bool)
          - update(success: bool)  # legacy; reward inferred as 1.0/0.0
        """
        # Back-compat: update(True/False) → treat arg as success flag
        if isinstance(episode_return, bool) and success is None:
            success = episode_return
            episode_return = 1.0 if episode_return else 0.0

        # Defaults if not provided
        if episode_return is None:
            episode_return = 0.0
        if success is None:
            # Conservative default: infer success from positive return
            success = bool(episode_return > 0)

        # Enqueue and recompute
        self._append(float(episode_return), bool(success))
        self._recalc()
        return self._d

    # Backwards-compat shim if callers use 'observe'
    def observe(self, episode_return: float, success: bool) -> float:
        self._append(float(episode_return), bool(success))
        self._recalc()
        return self._d

    @property
    def success_rate(self) -> float:
        """Recent success fraction over the window."""
        n = len(self._successes)
        return (sum(self._successes) / n) if n else 0.0

    @property
    def difficulty(self) -> float:
        """Current difficulty value."""
        return self._d

    # Historical alias some codebases use
    @property
    def value(self) -> float:
        return self._d

    def snapshot(self) -> Dict:
        mean, sd = (0.0, 0.0)
        if self._returns:
            mean = stats.fmean(self._returns)
            sd = stats.pstdev(self._returns) if len(self._returns) > 1 else 0.0
        return {
            "difficulty": self._d,
            "success_rate": self.success_rate,
            "return_mean": mean,
            "return_sd": sd,
            "n": len(self._returns),
        }

    def set_window(self, new_window: int) -> None:
        """Resize history windows safely."""
        new_window = max(1, int(new_window))
        self.window = new_window
        self._resize_deques(new_window)

    # ----------------------------- internals -----------------------------

    def _append(self, reward: float, success: bool) -> None:
        # Ensure deques respect the current window
        if self._returns.maxlen != self.window or self._successes.maxlen != self.window:
            self._resize_deques(self.window)
        self._returns.append(reward)
        self._successes.append(1 if success else 0)

    def _resize_deques(self, size: int) -> None:
        def _resize(dq: Deque, n: int) -> Deque:
            tmp = deque(dq, maxlen=n)
            return tmp
        self._returns = _resize(self._returns, size)
        self._successes = _resize(self._successes, size)

    def _recalc(self) -> None:
        if not self._successes:
            return
        sr = self.success_rate
        err = (self.target_success - sr)
        # ↓ If success is below target (err>0), LOWER difficulty
        k = 0.15
        self._d = float(min(self.max_diff, max(self.min_diff, self._d - k * err)))
