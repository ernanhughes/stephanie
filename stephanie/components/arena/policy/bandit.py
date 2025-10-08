# stephanie/arena/policy/bandit.py
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from stephanie.components.arena.plugins.interfaces import JobCtx

Key = Tuple[str, str, str]   # (task_type, topic, season_or_all)

class UCB1Policy:
    def __init__(self, plays: List[str], c: float = 1.4):
        self.c = c
        self.plays = plays
        self.N: Dict[Key, int] = defaultdict(int)             # pulls per key
        self.S: Dict[Tuple[Key, str], float] = defaultdict(float)  # sum rewards per (key, play)
        self.C: Dict[Tuple[Key, str], int] = defaultdict(int)      # count per (key, play)

    def _key(self, ctx: JobCtx) -> Key:
        return (ctx.task_type, ctx.topic, ctx.season or "all")

    def choose(self, ctx: JobCtx) -> str:
        key = self._key(ctx)
        n = self.N[key] + 1
        # explore each play once
        for p in self.plays:
            if self.C[(key, p)] == 0:
                return p
        # UCB1
        best, best_val = None, -1e9
        for p in self.plays:
            c = self.C[(key, p)]
            avg = self.S[(key, p)] / c
            bonus = self.c * math.sqrt(math.log(n) / c)
            val = avg + bonus
            if val > best_val:
                best, best_val = p, val
        return best or self.plays[0]

    def update(self, ctx: JobCtx, play: str, reward: float):
        key = self._key(ctx)
        self.N[key] += 1
        self.S[(key, play)] += reward
        self.C[(key, play)] += 1
