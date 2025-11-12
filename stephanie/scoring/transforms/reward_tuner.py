# stephanie/scoring/transforms/reward_tuner.py
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

log = logging.getLogger(__name__) 

class RewardRegistry:
    """
    Per-goal lightweight reward weight tuner (EMA). Keeps the "one reward"
    shape but lets weights drift by goal_id based on components of *validated*
    winners (k, c, g).
    """
    def __init__(self, memory: Any, container: Any, namespace: str = "arena.reward"):
        self.kv = getattr(memory, "kv", None) or getattr(container, "kv", None)
        self.ns = namespace

    def _key(self, goal_id: str) -> str:
        return f"{self.ns}:{goal_id or 'default'}"

    def weights(self, goal_id: str) -> Dict[str, float]:
        default = {"k": 0.6, "c": 0.25, "g": 0.15}
        try:
            w = self.kv.get(self._key(goal_id)) if self.kv else None
            if not isinstance(w, dict):
                return default
            # normalize defensively
            s = sum(max(0.0, float(v)) for v in w.values()) or 1.0
            return {k: max(0.0, float(v)) / s for k, v in w.items() if k in ("k","c","g")}
        except Exception:
            return default

    def update(self, goal_id: str, components: Tuple[float, float, float], alpha: float = 0.05):
        try:
            k, c, g = [max(0.0, min(1.0, float(x))) for x in components]
        except Exception:
            return
        w = self.weights(goal_id)
        # EMA toward observed components
        w["k"] = 0.95*w["k"] + alpha*k
        w["c"] = 0.95*w["c"] + alpha*c
        w["g"] = 0.95*w["g"] + alpha*g
        # normalize
        s = sum(w.values()) or 1.0
        w = {k: v/s for k,v in w.items()}
        try:
            if self.kv:
                self.kv.put(self._key(goal_id), w)
        except Exception as e:
            log.debug(f"RewardRegistry update failed for {goal_id}: {e}")
        return w