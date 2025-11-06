# stephanie/agents/policies/origin_policy.py
from __future__ import annotations

import logging
import random
from typing import Any, List, Tuple

log = logging.getLogger(__name__)

class _KV:
    """Tiny KV wrapper over memory.kv or container.get('kv')."""
    def __init__(self, memory: Any, container: Any, namespace: str = "arena.router"):
        self.mem = memory
        self.kv  = getattr(memory, "kv", None) or getattr(container, "kv", None)
        self.ns  = namespace

    def _key(self, key: str) -> str:
        return f"{self.ns}:{key}"

    def get(self, key: str, default):
        try:
            k = self._key(key)
            val = self.kv.get(k) if self.kv else None
            return val if val is not None else default
        except Exception:
            return default

    def put(self, key: str, value):
        try:
            k = self._key(key)
            if self.kv:
                self.kv.put(k, value)
        except Exception as e:
            log.debug(f"KV put failed for {key}: {e}")

class ThompsonOriginRouter:
    """
    Contextual Thompson Sampling over (origin, goal_id, domain) triples.
    We store Beta(a,b) parameters; update with a small credit depending on reward.
    """
    def __init__(self, memory: Any, container: Any, namespace: str = "arena.router"):
        self.kv = _KV(memory, container, namespace)

    def _ctx_key(self, origin: str, goal_id: str, domain: str) -> str:
        gid = goal_id or "default_goal"
        dom = (domain or "generic").lower()
        o   = (origin or "unknown").lower()
        return f"{o}::{gid}::{dom}"

    def recommend(self, origins: List[str], goal_id: str, domain: str, k: int, explore: float = 0.10) -> List[str]:
        draws: List[Tuple[str, float]] = []
        for o in origins:
            a, b = self.kv.get(self._ctx_key(o, goal_id, domain), (1.0, 1.0))
            sample = random.betavariate(max(1e-3, a), max(1e-3, b)) + explore*random.random()
            draws.append((o, sample))
        draws.sort(key=lambda t: t[1], reverse=True)
        return [o for o, _ in draws[:max(1, int(k))]]

    def update(self, origin: str, goal_id: str, domain: str, reward_overall: float):
        key = self._ctx_key(origin, goal_id, domain)
        a, b = self.kv.get(key, (1.0, 1.0))
        # credit shaping: map 0..1 → fractional success
        credit = 1.0 if reward_overall >= 0.85 else (0.5 if reward_overall >= 0.65 else 0.0)
        a = float(a) + credit
        b = float(b) + (1.0 - credit)
        self.kv.put(key, (a, b))
