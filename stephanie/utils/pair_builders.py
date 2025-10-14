# stephanie/scoring/utils/pair_builders.py
from __future__ import annotations
import torch

def make_balanced_pair(zA, zB, vA: float, vB: float, flip_p: float = 0.5, w_scale=0.05, w_cap=3.0, device="cpu"):
    delta = float(vA) - float(vB)
    if torch.rand(()) < flip_p:
        x = (zA - zB).squeeze(0).detach()
        y = 1.0 if delta > 0 else 0.0
    else:
        x = (zB - zA).squeeze(0).detach()
        y = 1.0 if delta < 0 else 0.0
    w = min(1.0 + w_scale*abs(delta), w_cap)
    return x, torch.tensor(y, dtype=torch.float32, device=device), torch.tensor(w, dtype=torch.float32, device=device)
