# stephanie/scoring/analysis/trace_tap.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class TraceSeries:
    name: str
    xs: List[torch.Tensor] = field(default_factory=list)

    def add(self, t: torch.Tensor) -> None:
        # detach + float + cpu to avoid autograd + VRAM growth
        self.xs.append(t.detach().float().cpu())

    def stack(self) -> torch.Tensor:
        # returns [T, B, D] typically
        return torch.stack(self.xs, dim=0) if self.xs else torch.empty(0)


class TraceTap:
    """
    Tiny in-memory collector for latent trajectories.
    Intended to live only in pipeline context (not persisted to DB).
    """
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._data: Dict[str, TraceSeries] = {}

    def add(self, name: str, t: torch.Tensor) -> None:
        if not self.enabled:
            return
        self._data.setdefault(name, TraceSeries(name=name)).add(t)

    def get(self, name: str) -> Optional[TraceSeries]:
        return self._data.get(name)

    def dump(self) -> Dict[str, torch.Tensor]:
        return {k: v.stack() for k, v in self._data.items()}

    def clear(self) -> None:
        self._data.clear()
