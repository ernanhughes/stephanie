"""
Adapters for EBT and VPM so JAS can run immediately.
Swap these with your real implementations via DI.
"""
from __future__ import annotations
import math
import random

import torch

class EBTAdapter:
    """
    Minimal adapter: .score(embedding)->[0..1]
    Replace with your real EBT (e.g., cosine/compatibility to goal/core vectors).
    """
    def __init__(self, dim: int = 1024, seed: int = 17):
        g = torch.Generator().manual_seed(seed)
        self.core = torch.randn(dim, generator=g)
        self.core = self.core / (self.core.norm() + 1e-8)

    def score(self, emb: torch.Tensor) -> float:
        emb = emb.view(-1)
        emb = emb / (emb.norm() + 1e-8)
        cos = float(torch.clamp(torch.dot(self.core, emb), -1, 1))
        # “Energy” as incompatibility → map cosine [-1,1] to [0,1]
        return 0.5 * (1 - cos)

class VPMAdapter:
    """
    Minimal adapter: .diversity()->[0..1], .mutate_rate (rw), .active_count()->int
    Replace with Memcube/your VPM manager.
    """
    def __init__(self):
        self.mutate_rate = 0.05
        self._active = 400

    def diversity(self) -> float:
        # simulate diversity oscillation
        return 0.6 + 0.3 * math.sin(random.random() * 3.14)

    def active_count(self) -> int:
        return self._active
