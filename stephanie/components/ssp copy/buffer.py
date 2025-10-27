# stephanie/components/ssp/buffer.py
from __future__ import annotations

from collections import deque
from typing import List, Dict, Any, Tuple

class EpisodeBuffer:
    def __init__(self, capacity=4096):
        self.buf = deque(maxlen=capacity)
    def add_trajectory(self, traj: List[Tuple[Any,Any,float]], meta: Dict):
        self.buf.append({"traj": traj, "meta": meta})
        return len(self.buf)-1
    def sample(self, n: int):
        n = min(n, len(self.buf))
        return list(list(self.buf)[-n:])
