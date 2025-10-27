from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional


class EpistemicRewardCalculator:
    """
    Intrinsic rewards for self-play:
      - learning_progress: magnitude of state change × verification score
      - novelty_bonus: 1 - max cosine similarity vs. recent solutions
      - final: blend of extrinsic (verification score) + intrinsic signals
    """

    def __init__(self, w_ext: float = 0.6, w_lp: float = 0.25, w_nov: float = 0.15):
        self.w_ext = float(w_ext)
        self.w_lp = float(w_lp)
        self.w_nov = float(w_nov)
        # a tiny ring-buffer of recent answers for novelty (strings or vectors)
        self._recent_text: list[str] = []

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / (na * nb))

    def learning_progress(self, before: np.ndarray, after: np.ndarray, verification_score: float) -> float:
        delta = float(np.linalg.norm(after - before))
        return delta * float(max(0.0, min(1.0, verification_score)))

    def novelty_bonus_text(self, answer: str) -> float:
        if not self._recent_text:
            self._recent_text.append(answer)
            return 1.0
        sims = []
        for prev in self._recent_text[-50:]:
            # crude Jaccard over tokens; replace with embedding cosine if available
            s1, s2 = set(answer.split()), set(prev.split())
            inter = len(s1 & s2)
            union = len(s1 | s2) or 1
            sims.append(inter / union)
        bonus = 1.0 - max(sims) if sims else 1.0
        self._recent_text.append(answer)
        return float(max(0.0, min(1.0, bonus)))

    def blend(
        self,
        verification_score: float,
        lp: Optional[float] = None,
        novelty: Optional[float] = None,
    ) -> float:
        lp = 0.0 if lp is None else lp
        novelty = 0.0 if novelty is None else novelty
        return float(
            self.w_ext * verification_score +
            self.w_lp * lp +
            self.w_nov * novelty
        )
