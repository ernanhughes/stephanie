# stephanie/scoring/model/hrm_model.py
from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Normalizes across features while preserving scale via a learned weight.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class RecurrentBlock(nn.Module):
    """
    A recurrent update block used by both L and H modules.
    Internally uses a GRUCell + RMSNorm for stable updates.
    """
    def __init__(self, input_dim: int, hidden_dim: int, name: str = "RecurrentBlock"):
        super().__init__()
        self.name = name
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, z_prev: torch.Tensor, input_combined: torch.Tensor) -> torch.Tensor:
        z_next = self.rnn_cell(input_combined, z_prev)
        z_next = self.norm(z_next)
        return z_next

    @staticmethod
    def init_state(batch_size: int, hidden_dim: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, hidden_dim, device=device)


class InputProjector(nn.Module):
    """
    Projects the input embedding into the HRM hidden space (x_tilde).
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.drop(self.project(x)))


class HRMModel(nn.Module):
    """
    Hierarchical Reasoning Model (HRM)

    Two coupled RNNs:
      - Low-level L: T fine-grained updates per cycle
      - High-level H: 1 abstract update per cycle

    Updated to mirror Tiny's diagnostic surface:
      - score_head (sigmoid with temperature)
      - logvar_head (aleatoric; used for certainty)
      - aux3_head (bad/mid/good logits → entropy/confidence)
      - disagree_head (proxy for |HRM - Tiny| learned only with HRM labels if available)
      - consistency_head (mask-invariance)
      - ood_head (in/out-of-distribution)
      - temp_head (calibration temperature)
      - recon_head (reconstruct x_tilde; comprehension proxy)
      - jacobian_fd (finite-diff sensitivity w.r.t. input)
    """

    def __init__(self, cfg: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__()
        self.logger = logger or _logger

        # Hyperparameters
        self.input_dim = int(cfg.get("input_dim", 2048))
        self.h_dim = int(cfg.get("h_dim", 256))
        self.l_dim = int(cfg.get("l_dim", 128))
        self.n_cycles = int(cfg.get("n_cycles", 4))
        self.t_steps = int(cfg.get("t_steps", 4))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.consistency_mask_p = float(cfg.get("consistency_mask_p", 0.10))
        self.fd_eps = float(cfg.get("fd_eps", 1e-3))

        # Device (will be overwritten by .to(device))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Input projection
        self.input_projector = InputProjector(self.input_dim, self.h_dim, dropout=self.dropout)

        # Recurrent modules
        self.l_module = RecurrentBlock(2 * self.h_dim, self.l_dim, name="LModule")
        self.h_module = RecurrentBlock(self.l_dim + self.h_dim, self.h_dim, name="HModule")

        # Heads (operate on zH)
        self.head_drop = nn.Dropout(self.dropout)
        self.score_head = nn.Linear(self.h_dim, 1)         # sigmoid(score_logit / tau)
        self.logvar_head = nn.Linear(self.h_dim, 1)        # aleatoric log-variance
        self.aux3_head = nn.Linear(self.h_dim, 3)          # bad/mid/good
        self.disagree_head = nn.Linear(self.h_dim, 1)      # sigmoid
        self.consistency_head = nn.Linear(self.h_dim, 1)   # sigmoid
        self.ood_head = nn.Linear(self.h_dim, 1)           # sigmoid
        self.temp_head = nn.Linear(self.h_dim, 1)          # softplus→tau
        self.recon_head = nn.Linear(self.h_dim, self.h_dim)  # reconstruct x_tilde

        # Final norm on zH (stabilize heads)
        self.final_norm = RMSNorm(self.h_dim)

    # ---------------------------
    # Core rollout
    # ---------------------------
    def _rollout(self, x_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the hierarchical recurrence; returns (zL_final, zH_final, zH_traj_max) where zH_traj_max
        is an aggregation used to emulate a simple "halting-like" evidence accumulation if needed.
        """
        B = x_tilde.size(0)
        zL = RecurrentBlock.init_state(B, self.l_dim, self.device)
        zH = RecurrentBlock.init_state(B, self.h_dim, self.device)
        zH_max = torch.zeros_like(zH)

        for _ in range(self.n_cycles):
            for _t in range(self.t_steps):
                l_input = torch.cat([x_tilde, zH], dim=-1)  # (B, 2*h_dim)
                zL = self.l_module(zL, l_input)

            h_input = torch.cat([zL, zH], dim=-1)          # (B, l_dim + h_dim)
            zH = self.h_module(zH, h_input)
            zH_max = torch.maximum(zH_max, zH)

        zH = self.final_norm(zH)
        zH_max = self.final_norm(zH_max)
        return zL, zH, zH_max

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(
        self,
        x: torch.Tensor,                      # (B, input_dim) — embedding of (goal⊕response) or plan
        *,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:
          score01: (B,1) in [0,1]
          aux: dict of raw/derived diagnostics
        """
        B = x.size(0)

        # Project input and roll out recurrence
        x_tilde = self.input_projector(x)     # (B, h_dim)
        zL, zH, zH_max = self._rollout(x_tilde)
        zH_head = self.head_drop(zH)

        # Temperature & primary score
        tau = 0.5 + 0.5 * F.softplus(self.temp_head(zH_head))     # τ ∈ (0.5, ∞)
        score_logit = self.score_head(zH_head)
        score01 = torch.sigmoid(score_logit / tau)

        # Other heads
        log_var = self.logvar_head(zH_head)
        aux3_logits = self.aux3_head(zH_head)
        aux3_probs = F.softmax(aux3_logits, dim=-1)
        disagree_hat = torch.sigmoid(self.disagree_head(zH_head))
        ood_hat = torch.sigmoid(self.ood_head(zH_head))

        # Consistency target (mask-invariance in head-space)
        mask = (torch.rand_like(zH_head) < self.consistency_mask_p).float()
        zH_masked = zH_head * (1.0 - mask)
        consistency_hat = torch.sigmoid(self.consistency_head(zH_head))
        consistency_target = self._cos01(zH_head, zH_masked).unsqueeze(-1)

        # Reconstruction of x_tilde (comprehension proxy)
        x_recon = self.recon_head(zH_head)
        recon_sim = self._cos01(x_recon, x_tilde).unsqueeze(-1)

        # Finite-difference sensitivity wrt input x
        x_eps = x + self.fd_eps * F.normalize(torch.randn_like(x), dim=-1)
        with torch.no_grad():
            x_tilde_eps = self.input_projector(x_eps)
            _, zH_eps, _ = self._rollout(x_tilde_eps)
            zH_eps = self.head_drop(zH_eps)
            tau_eps = 0.5 + 0.5 * F.softplus(self.temp_head(zH_eps))
            score_eps = torch.sigmoid(self.score_head(zH_eps) / tau_eps)
        jacobian_fd = ((score_eps - score01).abs() / self.fd_eps).clamp(0, 10.0) / 10.0

        # "Halting-like" signal (no true halting; aggregate evidence from zH_max)
        halt_logit = (zH_max * zH_head).mean(-1, keepdim=True) / max(self.h_dim, 1)  # scaled dot
        halt_prob = torch.sigmoid(halt_logit)

        if not return_aux:
            return score01, {}

        aux: Dict[str, Any] = {
            # Raw heads
            "score_logit": score_logit,                 # (B,1)
            "log_var": log_var,                         # (B,1)
            "aux3_logits": aux3_logits,                 # (B,3)
            "disagree_logit": self.disagree_head(zH_head),  # (B,1)
            "consistency_logit": self.consistency_head(zH_head),  # (B,1)
            "x_recon": x_recon,                         # (B,h_dim)

            # Derived / heat-mappable
            "score": score01,                           # (B,1) in [0,1]
            "certainty01": torch.sigmoid(-log_var),     # higher = more certain
            "uncertainty": torch.sigmoid(-log_var),     # alias for back-compat
            "aux3_probs": aux3_probs,                   # (B,3)
            "entropy_aux": (-(aux3_probs * F.log_softmax(aux3_logits, -1)).sum(-1)
                             / torch.log(torch.tensor(3.0, device=x.device))).unsqueeze(-1),  # (B,1)
            "disagree_hat": disagree_hat,               # (B,1)
            "consistency_hat": consistency_hat,         # (B,1)
            "consistency_target": consistency_target,   # (B,1)
            "recon_sim": recon_sim,                     # (B,1)
            "ood_hat": ood_hat,                         # (B,1)
            "temp01": (torch.tanh(tau) + 1) / 2,        # proxy (B,1) ∈ (0,1)
            "jacobian_fd": jacobian_fd,                 # (B,1)
            "halt_prob": halt_prob,                     # (B,1) pseudo-halting
            # Introspection
            "zL_final": zL,
            "zH_final": zH,
        }
        return score01, aux

    # ---------------------------
    # Utils
    # ---------------------------
    @staticmethod
    def _cos01(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """Cosine similarity mapped from [-1,1] to [0,1]."""
        sim = F.cosine_similarity(a, b, dim=dim, eps=eps)
        return (sim + 1.0) * 0.5

    def to(self, device):
        super().to(device)
        self.device = device
        return self
