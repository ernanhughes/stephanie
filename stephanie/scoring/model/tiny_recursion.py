
"""
Tiny Recursion Model (Tiny+) - Parameter-Efficient Recursive Neural Architecture

This module implements a compact, recursive neural network for multi-task evaluation
of AI model responses. The architecture combines recursive state updates with
multi-head output predictions, enabling efficient quality assessment across
multiple dimensions from embedding inputs.

Key Innovations:
- Recursive latent state updates with halting mechanisms
- Sparse Autoencoder (SAE) bottleneck for interpretable concepts
- Multi-head prediction for comprehensive quality assessment
- Heteroscedastic uncertainty estimation
- In-graph consistency regularization

Architecture Overview:
1. Recursive fusion of goal (x), response (y), and latent (z) states
2. Core processing blocks (attention or MLP-based)
3. SAE bottleneck for sparse concept representation
4. Multi-head prediction for scores, uncertainty, and auxiliary tasks

Author: Stephanie AI Team
Version: 2.0
Date: 2024
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Core Building Blocks
# ---------------------------

class TinyBlock(nn.Module):
    """
    Basic residual block: LayerNorm → MLP → residual connection.

    Supports both 2D [batch, features] and 3D [batch, sequence, features] inputs.
    Uses GELU activation and dropout for regularization.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # Expansion factor 4
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),  # Projection back
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block: x + MLP(LayerNorm(x))"""
        return x + self.mlp(self.ln(x))


class TinyBlockAttn(nn.Module):
    """
    Attention-enhanced residual block with Multi-Head Self-Attention.

    Architecture: LN → MHA → residual → TinyBlock → residual
    Automatically handles 2D/3D inputs and returns same dimensionality.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # [batch, seq, features]
        )
        self.drop = nn.Dropout(dropout)
        self.ff = TinyBlock(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic shape handling.

        Args:
            x: Input tensor of shape [B, D] or [B, L, D]

        Returns:
            Output tensor with same shape as input
        """
        squeeze_back = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] → [B, 1, D]
            squeeze_back = True

        q = k = v = self.ln_attn(x)
        h, _ = self.attn(q, k, v, need_weights=False)
        x = x + self.drop(h)  # Residual connection
        x = self.ff(x)        # Feed-forward with residual

        if squeeze_back:
            x = x.squeeze(1)  # [B, 1, D] → [B, D]
        return x


# ---------------------------
# Tiny Recursion Model (Tiny+)
# ---------------------------

class TinyRecursionModel(nn.Module):
    """
    Parameter-efficient recursive model for multi-task evaluation.

    Recursively updates latent state z using goal (x) and response (y) embeddings
    over multiple steps. Features comprehensive multi-head prediction and
    sparse autoencoder bottleneck for interpretable representations.

    Core Components:
    - Recursive state fusion: [x, y, z] → z'
    - Core processing stack: Attention or MLP blocks
    - SAE bottleneck: Sparse concept encoding
    - Multi-head prediction: 12 specialized output heads

    Inputs:
        x: Goal/condition embedding [B, D]
        y: Response embedding [B, D]
        z: Initial latent state [B, D] (typically zeros)

    Outputs:
        logits: Classification logits [B, vocab_size] (legacy compatibility)
        halt_logits: Halting signal logits [B]
        z_final: Final latent state after recursion [B, D]
        aux: Dictionary of auxiliary predictions and metrics
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
        step_scale: float = 0.1,           # Residual scaling for state updates
        consistency_mask_p: float = 0.10,  # Mask probability for consistency regularization
        len_norm_L: float = 512.0,         # Length normalization constant
        enable_agree_head: bool = True,    # Enable agreement prediction head
        enable_causal_sens_head: bool = True,  # Enable sensitivity prediction head
    ):
        super().__init__()

        # Model configuration
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

        # Core processing stack
        if use_attention:
            blocks = [TinyBlockAttn(d_model, n_heads=attn_heads, dropout=dropout)
                      for _ in range(n_layers)]
        else:
            blocks = [TinyBlock(d_model, dropout=dropout) for _ in range(n_layers)]
        self.core = nn.Sequential(*blocks)

        # State fusion: combine goal, response, and latent states
        self.z_proj = nn.Linear(d_model * 3, d_model)  # [x, y, z] → z'
        self.final_ln = nn.LayerNorm(d_model)

        # Core prediction heads
        self.halt_head = nn.Linear(d_model, 1)            # Halting signal logits
        self.classifier = nn.Linear(d_model, vocab_size)  # Legacy classification

        # Extended prediction heads
        self.score_head = nn.Linear(d_model, 1)        # Quality score ∈ [0,1]
        self.logvar_head = nn.Linear(d_model, 1)       # Aleatoric uncertainty (log-variance)
        self.aux3_head = nn.Linear(d_model, 3)         # 3-way classification
        self.disagree_head = nn.Linear(d_model, 1)     # Disagreement prediction
        self.recon_head = nn.Linear(d_model, d_model)  # Embedding reconstruction
        self.consistency_head = nn.Linear(d_model, 1)  # Robustness prediction
        self.ood_head = nn.Linear(d_model, 1)          # OOD detection
        self.temp_head = nn.Linear(d_model, 1)         # Temperature calibration

        # Bridge heads
        self.agree_head = nn.Linear(d_model, 1)        # Cross-model agreement
        self.causal_sens_head = nn.Linear(d_model, 1)  # Perturbation sensitivity

        # Sparse Autoencoder (SAE) bottleneck
        self.sae_enc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # Compression
            nn.ReLU(),
            nn.LayerNorm(d_model // 2),
        )
        self.sae_dec = nn.Linear(d_model // 2, d_model)  # Reconstruction
        self.sae_alpha = 0.05  # SAE reconstruction loss weight

        # Regularization
        self.head_drop = nn.Dropout(dropout)

    @staticmethod
    def _cos01(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute cosine similarity mapped from [-1, 1] to [0, 1].

        Args:
            a, b: Input tensors to compare
            dim: Dimension for cosine computation
            eps: Numerical stability term

        Returns:
            Cosine similarity in range [0, 1] where 1 = identical
        """
        sim = F.cosine_similarity(a, b, dim=dim, eps=eps)
        return (sim + 1.0) * 0.5

    def _recur(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute recursive state updates over n_recursions steps.

        Process:
          1. Fuse [x, y, z] → z_next via projection and activation
          2. Process through core network stack
          3. Update halting signals
          4. Apply residual state update: z = z + step_scale * z_next
          5. Apply SAE bottleneck to final state

        Args:
            x: Goal embedding [B, D]
            y: Response embedding [B, D]
            z: Initial latent state [B, D]

        Returns:
            z_final: Final latent state after recursion [B, D]
            z_head: SAE-processed state for prediction heads [B, D]
            halt_logits: Maximum halting logits across steps [B, 1]
            tau: Temperature parameter for score calibration [B, 1]
            c: Sparse concept codes from SAE bottleneck [B, D//2]
        """
        B = x.size(0)
        device = x.device

        # Initialize halting signals to very negative values
        halt_logits = torch.full((B, 1), -1e9, device=device)
        z_cur = z  # Current latent state

        # Recursive state updates
        for _ in range(self.n_recursions):
            fused = torch.cat([x, y, z_cur], dim=-1)   # [B, 3 * D]
            z_next = torch.tanh(self.z_proj(fused))    # [B, D] with saturation
            z_next = self.core(z_next)                 # [B, D] core processing

            # Update halting signal (track maximum across steps)
            step_halt = self.halt_head(self.final_ln(z_next))  # [B, 1]
            halt_logits = torch.maximum(halt_logits, step_halt)

            # Residual state update with step scaling
            z_cur = z_cur + self.step_scale * z_next

        # Final normalization
        z_final = self.final_ln(z_cur)  # [B, D]

        # Sparse Autoencoder bottleneck
        c = self.sae_enc(z_final)                  # [B, D//2] concept codes
        z_head = z_final + self.sae_dec(c)         # [B, D] with SAE reconstruction
        z_head = self.head_drop(z_head)            # Regularization

        # Temperature calibration parameter (τ ∈ (0.5, ∞))
        tau_raw = self.temp_head(z_head)
        tau = 0.5 + 0.5 * F.softplus(tau_raw)  # Lower bound at 0.5

        return z_final, z_head, halt_logits, tau, c

    def forward(
        self,
        x: torch.Tensor,                    # Goal embedding [B, D]
        y: torch.Tensor,                    # Response embedding [B, D]
        z: torch.Tensor,                    # Initial latent state [B, D]
        *,
        seq_len: Optional[torch.Tensor] = None,  # Response length [B] (optional)
        return_aux: bool = True,                 # Whether to return auxiliary outputs
        with_consistency_target: bool = True,    # Compute consistency regularization
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Complete forward pass with recursive processing and multi-head prediction.
        """
        # Main recursive processing
        z = z.clone()  # Ensure we don't modify input
        z_final, z_head, halt_logits, tau, c = self._recur(x, y, z)

        # Core prediction heads
        logits = self.classifier(z_head)                    # [B, vocab_size]
        score_logit = self.score_head(z_head)               # [B, 1]
        log_var = self.logvar_head(z_head)                  # [B, 1] uncertainty

        # ----- NUMERICAL SAFETY -----
        LOGVAR_MIN, LOGVAR_MAX = -5.0, 5.0
        log_var = log_var.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)

        # Use tau for calibration; keep a stable proxy for telemetry
        # NOTE: move temp01 to sigmoid(tau_raw) for cross-model alignment
        tau_raw = self.temp_head(z_head) 
        tau = 0.5 + 0.5 * F.softplus(tau_raw)
        tau_safe = torch.clamp(tau, min=1e-2)
        s = torch.sigmoid(score_logit / tau_safe)

        # ----- Core auxiliaries
        aux3_logits = self.aux3_head(z_head)
        aux3_probs  = F.softmax(aux3_logits, dim=-1)
        disagree_logit = self.disagree_head(z_head)
        y_recon     = self.recon_head(z_head)
        ood_logit   = self.ood_head(z_head)

        # Optional bridge heads
        agree01 = torch.sigmoid(self.agree_head(z_head)) if self.enable_agree_head else None
        sens01  = torch.sigmoid(self.causal_sens_head(z_head)) if self.enable_causal_sens_head else None

        # Consistency target 
        mask = (torch.rand_like(z_head) < self.consistency_mask_p).float()
        z_masked = z_head * (1.0 - mask)
        cos_consistency = self._cos01(z_head, z_masked).unsqueeze(-1)
        consistency_logit = self.consistency_head(z_head)

        # Finite-difference sensitivity
        eps = 1e-3
        y_eps = y + eps * F.normalize(torch.randn_like(y), dim=-1)
        with torch.no_grad():
            _, z_head_eps, _, tau_eps, _ = self._recur(x, y_eps, z)
        tau_eps_safe = torch.clamp(tau_eps, min=1e-2)
        score_eps    = torch.sigmoid(self.score_head(z_head_eps) / tau_eps_safe)
        jac_fd       = ((score_eps - s).abs() / eps).clamp(0, 10.0) / 10.0

        # Length effect
        if seq_len is not None:
            len_effect = torch.tanh((seq_len.float() / self.len_norm_L)).unsqueeze(-1)
        else:
            len_effect = torch.zeros_like(s)
        length_norm01 = (len_effect + 1.0) * 0.5

        # ----- Aligned telemetry keys -----
        certainty01   = torch.sigmoid(-log_var)
        uncertainty01 = 1.0 - certainty01
        temp01        = torch.sigmoid(tau_raw)  # aligned proxy in [0,1]
        ood_hat01     = torch.sigmoid(ood_logit)
        halt_prob     = torch.sigmoid(halt_logits).unsqueeze(-1) if halt_logits.dim()==1 else torch.sigmoid(halt_logits)

        # Device-safe normalized entropy (in [0,1])
        logK = torch.log(torch.tensor(3.0, device=z_head.device, dtype=z_head.dtype))
        entropy_aux = (-(aux3_probs * F.log_softmax(aux3_logits, -1)).sum(-1) / logK).unsqueeze(-1)

        aux: Dict[str, Any] = {
            # raw heads you need for training
            "score_logit": score_logit,
            "log_var": log_var,
            "aux3_logits": aux3_logits,
            "disagree_logit": disagree_logit,
            "y_recon": y_recon,
            "consistency_logit": consistency_logit,
            "consistency_target": cos_consistency.detach(),

            # aligned derived telemetry (all ∈ [0,1])
            "score": s,
            "certainty01": certainty01,
            "uncertainty01": uncertainty01,     # <— NEW (correct)
            "uncertainty": uncertainty01,       # <— OPTIONAL alias for back-compat
            "aux3_probs": aux3_probs,
            "entropy_aux": entropy_aux,
            "disagree_hat": torch.sigmoid(disagree_logit),
            "recon_sim": self._cos01(y_recon, y).unsqueeze(-1),
            "consistency_hat": torch.sigmoid(consistency_logit),
            "concept_sparsity": (c > 0).float().mean(dim=-1, keepdim=True),
            "ood_hat01": ood_hat01,             # <— NEW aligned name
            "temp01": temp01,                   # <— changed to sigmoid(tau_raw)
            "jacobian_fd": jac_fd,
            "len_effect": len_effect,
            "length_norm01": length_norm01,     # <— NEW 0..1 length proxy
            "halt_prob": halt_prob,             # <— NEW
        }

        if agree01 is not None:
            aux["agree01"] = agree01
        if sens01 is not None:
            aux["sens01"] = sens01

        return logits, halt_logits.squeeze(-1), z_final, (aux if return_aux else {})
