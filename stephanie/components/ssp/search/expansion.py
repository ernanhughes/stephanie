from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple


class StrategicExpansion:
    """
    Basic UCB-style node selector mixing exploitation (avg reward) and exploration (visits).
    Works over a dictionary: node_id -> (avg_reward, visits).
    """

    def __init__(self, c: float = 1.25):
        self.c = float(c)
        self.total_visits = 1

    def select(self, stats: Dict[str, Tuple[float, int]], k: int = 1) -> List[str]:
        """
        stats: {node_id: (avg_reward, visits)}
        returns: top-k node_ids by UCB score
        """
        if not stats:
            return []
        self.total_visits = max(self.total_visits, sum(v for _, v in stats.values()))
        scored: List[Tuple[str, float]] = []
        for nid, (avg, n) in stats.items():
            n = max(1, int(n))
            bonus = self.c * math.sqrt(math.log(self.total_visits + 1) / n)
            scored.append((nid, float(avg + bonus)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scored[:k]]
