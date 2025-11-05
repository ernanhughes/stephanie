from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class VPMThoughtModelConfig:
    in_channels: int = 3
    hidden_dim: int = 512
    goal_dim: int = 128
    n_ops: int = 5
    param_dim: int = 8
    dropout: float = 0.1

class VPMSpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(512, out_dim), nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))

class VPMThoughtPolicy(nn.Module):
    """
    Stephanie-native wrapper around your thought policy.
    API: forward(vpm[B,C,H,W], goal_vec[B,G]) -> (op_logits[B,K], param_mean[B,D], param_log_std[B,D], value[B,1])
    """
    def __init__(self, cfg: VPMThoughtModelConfig):
        super().__init__()
        self.cfg = cfg
        C, H, G = cfg.in_channels, cfg.hidden_dim, cfg.goal_dim
        self.encoder = VPMSpatialEncoder(C, H)
        self.goal_proj = nn.Sequential(nn.Linear(C, G), nn.ReLU(), nn.Linear(G, G))
        self.fuser = nn.Sequential(nn.Linear(H + G, H), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.op_head = nn.Linear(H, cfg.n_ops)
        self.param_head = nn.Linear(H, cfg.param_dim * 2)
        self.value_head = nn.Linear(H, 1)

    def forward(self, vpm: torch.Tensor, goal_vec: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        s = self.encoder(vpm)
        g = self.goal_proj(goal_vec)
        h = self.fuser(torch.cat([s, g], dim=-1))
        op_logits = self.op_head(h)
        pr = self.param_head(h)
        D = self.cfg.param_dim
        param_mean = torch.tanh(pr[:, :D])
        param_log_std = pr[:, D:]
        value = self.value_head(h)
        return op_logits, param_mean, param_log_std, value
