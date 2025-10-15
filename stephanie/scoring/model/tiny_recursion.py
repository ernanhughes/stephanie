# stephanie/models/tiny_recursion.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Core Blocks
# ---------------------------

class TinyBlock(nn.Module):
    """LN → MLP → residual; works with [B, D] or [B, L, D]."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln(x))


class TinyBlockAttn(nn.Module):
    """
    LN → MHA → residual, then TinyBlock.
    Accepts [B, D] or [B, L, D]; returns same rank as input.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.ff = TinyBlock(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_back = False
        if x.dim() == 2:  # [B, D] → [B, 1, D] for attention
            x = x.unsqueeze(1)
            squeeze_back = True

        q = k = v = self.ln_attn(x)
        h, _ = self.attn(q, k, v, need_weights=False)
        x = x + self.drop(h)  # residual
        x = self.ff(x)        # LN-MLP residual

        if squeeze_back:
            x = x.squeeze(1)  # [B, 1, D] → [B, D]
        return x


# ---------------------------
# Tiny Recursion (Tiny+)
# ---------------------------

class TinyRecursionModel(nn.Module):
    """
    Parameter-efficient recursive model over embeddings.
    Recurrently updates latent state z using (x=goal, y=response) for n_recursions.

    Heads:
      - classifier: optional token/vocab head (unchanged)
      - halt_head:  halting LOGIT across steps (take sigmoid at use time)
      - score_head: regression score in [0,1] (via sigmoid/temperature)
      - logvar_head: aleatoric log-variance for heteroscedastic loss
      - aux3_head:  3-way logits (bad/mid/good) → entropy/confidence
      - disagree_head: predicts |HRM - Tiny| in [0,1] (via sigmoid) using HRM label only at train
      - recon_head: reconstruct y embedding (cos/MSE loss) for comprehension signal
      - consistency_head: predicts cos(z, z_masked) ∈ [0,1] (robustness/invariance)
      - ood_head: in-/out-of-distribution score
      - temp_head: temperature for score calibration
      - agree_head: predicts cross-model agreement ∈ [0,1] (new)
      - causal_sens_head: predicts perturbation sensitivity ∈ [0,1] (new)
      - SAE (sae_enc/sae_dec): sparse concept bottleneck before heads

    Returns:
      logits:        [B, vocab_size] (class logits; compat)
      halt_logits:   [B] (max over steps)
      z_final:       [B, D] final latent after recursion and LN
      aux:           dict of raw head outputs + derived metrics
    """
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_recursions: int = 6,
        vocab_size: int = 1024,
        use_attention: bool = False,
        dropout: float = 0.1,
        attn_heads: int = 4,
        step_scale: float = 0.1,          # residual step factor for z updates
        consistency_mask_p: float = 0.10, # in-graph mask prob for consistency target
        len_norm_L: float = 512.0,        # length normalization constant for len_effect
        enable_agree_head: bool = True,          # predict cross-model agreement
        enable_causal_sens_head: bool = True,    # predict causal sensitivity (|Δscore|)
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_recursions = n_recursions
        self.vocab_size = vocab_size
        self.use_attention = use_attention
        self.step_scale = step_scale
        self.consistency_mask_p = consistency_mask_p
        self.len_norm_L = float(len_norm_L)
        self.enable_agree_head = enable_agree_head
        self.enable_causal_sens_head = enable_causal_sens_head

        # Core block stack
        if use_attention:
            blocks = [TinyBlockAttn(d_model, n_heads=attn_heads, dropout=dropout) for _ in range(n_layers)]
        else:
            blocks = [TinyBlock(d_model, dropout=dropout) for _ in range(n_layers)]
        self.core = nn.Sequential(*blocks)

        # Fusion
        self.z_proj = nn.Linear(d_model * 3, d_model)  # fuse [x, y, z] → z'
        self.final_ln = nn.LayerNorm(d_model)

        # Halting + classifier (legacy / optional)
        self.halt_head   = nn.Linear(d_model, 1)         # returns LOGITS
        self.classifier  = nn.Linear(d_model, vocab_size)

        # Tiny+ heads
        self.score_head       = nn.Linear(d_model, 1)        # sigmoid → [0,1] (with temperature)
        self.logvar_head      = nn.Linear(d_model, 1)        # aleatoric log-variance
        self.aux3_head        = nn.Linear(d_model, 3)        # bad/mid/good logits
        self.disagree_head    = nn.Linear(d_model, 1)        # sigmoid → [0,1]
        self.recon_head       = nn.Linear(d_model, d_model)  # y reconstruction in embedding space
        self.consistency_head = nn.Linear(d_model, 1)        # predict cos(z, z_masked) ∈ [0,1]
        self.ood_head         = nn.Linear(d_model, 1)        # OOD probability
        self.temp_head        = nn.Linear(d_model, 1)        # temperature

        # New bridge heads
        self.agree_head       = nn.Linear(d_model, 1)        # cross-model agreement ∈ [0,1]
        self.causal_sens_head = nn.Linear(d_model, 1)        # perturbation sensitivity ∈ [0,1]

        # SAE bottleneck
        self.sae_enc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model // 2),
        )
        self.sae_dec = nn.Linear(d_model // 2, d_model)
        self.sae_alpha = 0.05  # if you add recon loss for SAE

        # Head dropout
        self.head_drop = nn.Dropout(dropout)

    @staticmethod
    def _cos01(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """Cosine similarity mapped from [-1,1] to [0,1]."""
        sim = F.cosine_similarity(a, b, dim=dim, eps=eps)
        return (sim + 1.0) * 0.5

    def _recur(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the same n_recursions loop and return (z_final, z_head, halt_logits, tau, c).
        """
        B = x.size(0)
        device = x.device
        halt_logits = torch.full((B, 1), -1e9, device=device)
        z_cur = z

        for _ in range(self.n_recursions):
            fused = torch.cat([x, y, z_cur], dim=-1)   # [B, 3D]
            z_next = torch.tanh(self.z_proj(fused))    # [B, D]
            z_next = self.core(z_next)                 # [B, D]
            step_halt = self.halt_head(self.final_ln(z_next))  # [B, 1]
            halt_logits = torch.maximum(halt_logits, step_halt)
            z_cur = z_cur + self.step_scale * z_next

        z_final = self.final_ln(z_cur)                 # [B, D]

        # SAE bottleneck → z_head used for all heads
        c = self.sae_enc(z_final)                      # [B, C]
        z_head = z_final + self.sae_dec(c)             # [B, D]
        z_head = self.head_drop(z_head)

        # Temperature (calibration)
        tau = 0.5 + 0.5 * F.softplus(self.temp_head(z_head))  # τ ∈ (0.5, ∞)
        return z_final, z_head, halt_logits, tau, c

    def forward(
        self,
        x: torch.Tensor,            # [B, D] goal/condition embedding
        y: torch.Tensor,            # [B, D] response embedding
        z: torch.Tensor,            # [B, D] latent (can start at zeros)
        *,
        seq_len: Optional[torch.Tensor] = None,  # [B] token counts (optional)
        return_aux: bool = True,
        with_consistency_target: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward recursion.
        """
        # Main recurrence
        z = z.clone()
        z_final, z_head, halt_logits, tau, c = self._recur(x, y, z)

        # Heads from z_head
        logits         = self.classifier(z_head)                   # [B, vocab]
        score_logit    = self.score_head(z_head)                   # [B,1]
        s              = torch.sigmoid(score_logit / tau)          # calibrated score in [0,1]
        log_var        = self.logvar_head(z_head)                  # [B,1]
        aux3_logits    = self.aux3_head(z_head)                    # [B,3]
        aux3_probs     = F.softmax(aux3_logits, dim=-1)            # [B,3]
        disagree_logit = self.disagree_head(z_head)                # [B,1]
        y_recon        = self.recon_head(z_head)                   # [B,D]
        ood_logit      = self.ood_head(z_head)                     # [B,1]

        # New bridge heads
        if self.enable_agree_head:
            agree01 = torch.sigmoid(self.agree_head(z_head))       # [B,1]
        else:
            agree01 = None
        if self.enable_causal_sens_head:
            sens01  = torch.sigmoid(self.causal_sens_head(z_head)) # [B,1]
        else:
            sens01 = None

        # Cheap, in-graph consistency target (no extra forward)
        if with_consistency_target:
            mask = (torch.rand_like(z_head) < self.consistency_mask_p).float()
            z_masked = z_head * (1.0 - mask)
            cos_consistency = self._cos01(z_head, z_masked).unsqueeze(-1)  # [B,1]
        else:
            cos_consistency = torch.ones_like(s)  # [B,1]
        consistency_logit = self.consistency_head(z_head)          # [B,1]

        # Finite-difference sensitivity wrt y (full recurrence path)
        eps = 1e-3
        y_eps = y + eps * F.normalize(torch.randn_like(y), dim=-1)
        with torch.no_grad():
            _, z_head_eps, _, tau_eps, _ = self._recur(x, y_eps, z)
        score_eps = torch.sigmoid(self.score_head(z_head_eps) / tau_eps)
        jac_fd = ((score_eps - s).abs() / eps).clamp(0, 10.0) / 10.0  # [B,1] ∈ [0,1]

        # Optional len_effect
        if seq_len is not None:
            len_effect = torch.tanh((seq_len.float() / self.len_norm_L)).unsqueeze(-1)  # [B,1]
        else:
            len_effect = torch.zeros_like(s)

        # Aux dict
        aux: Dict[str, Any] = {
            # raw outputs
            "score_logit":       score_logit,                  # [B,1]
            "log_var":           log_var,                      # [B,1]
            "aux3_logits":       aux3_logits,                  # [B,3]
            "disagree_logit":    disagree_logit,               # [B,1]
            "y_recon":           y_recon,                      # [B,D]
            "consistency_logit": consistency_logit,            # [B,1]
            "consistency_target": cos_consistency.detach(),  # [B,1]
            # derived, heat-mappable (0..1 where sensible)
            "score":             s,                                        # [B,1]
            # NOTE: for historical compatibility we keep key "uncertainty" but it actually carries certainty.
            "certainty01":       torch.sigmoid(-log_var),                  # higher = more certain
            "uncertainty":       torch.sigmoid(-log_var),                  # alias of certainty01 (back-compat)
            "aux3_probs":        aux3_probs,                               # [B,3]
            "entropy_aux":       (-(aux3_probs * F.log_softmax(aux3_logits, -1)).sum(-1)
                                   / math.log(3.0)).unsqueeze(-1),         # [B,1] ∈ [0,1]
            "disagree_hat":      torch.sigmoid(disagree_logit),            # [B,1]
            "recon_sim":         self._cos01(y_recon, y).unsqueeze(-1),    # [B,1]
            "consistency_hat":   torch.sigmoid(consistency_logit),         # [B,1]
            "concept_vec":       c,                                        # [B,C]
            "concept_sparsity":  (c > 0).float().mean(dim=-1, keepdim=True),  # [B,1]
            "ood_hat":           torch.sigmoid(ood_logit),                 # [B,1]
            "temp01":            (torch.tanh(tau) + 1) / 2,                # [B,1] proxy in 0..1
            "jacobian_fd":       jac_fd,                                   # [B,1]
            "len_effect":        len_effect,                               # [B,1]
        }
        if agree01 is not None:
            aux["agree01"] = agree01
        if sens01 is not None:
            aux["sens01"] = sens01

        return logits, halt_logits.squeeze(-1), z_final, (aux if return_aux else {})
