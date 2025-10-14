# stephanie/models/tiny_recursion.py
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


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

        # Attn block with pre-norm + residual
        h, _ = self.attn(self.ln_attn(x), self.ln_attn(x), self.ln_attn(x), need_weights=False)
        x = x + self.drop(h)  # residual
        x = self.ff(x)        # LN-MLP residual

        if squeeze_back:
            x = x.squeeze(1)  # [B, 1, D] → [B, D]
        return x


class TinyRecursionModel(nn.Module):
    """
    Parameter-efficient recursive model over embeddings.
    Recurrently updates latent state z using (x=goal, y=response) for n_recursions,
    producing class logits and a halting **logit** suitable for BCEWithLogits.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.n_recursions = n_recursions
        self.vocab_size = vocab_size
        self.use_attention = use_attention

        # Core block stack
        if use_attention:
            blocks = [TinyBlockAttn(d_model, n_heads=attn_heads, dropout=dropout) for _ in range(n_layers)]
        else:
            blocks = [TinyBlock(d_model, dropout=dropout) for _ in range(n_layers)]
        self.core = nn.Sequential(*blocks)

        # Fusion + heads
        self.z_proj = nn.Linear(d_model * 3, d_model)  # fuse [x, y, z] → z'
        self.final_ln = nn.LayerNorm(d_model)
        self.halt_head = nn.Linear(d_model, 1)         # returns LOGITS
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,  # [B, D]
        y: torch.Tensor,  # [B, D]
        z: torch.Tensor,  # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:     [B, vocab_size]  (class logits)
            halt_logits:[B]              (max halting LOGIT across recursions)
            new_z:      [B, D]
        """
        B = x.size(0)
        device = x.device

        z = z.clone()
        # Accumulate halting **logits** via max (sigmoid is monotonic: max(logit) ↔ max(prob))
        halt_logits = torch.full((B, 1), -1e9, device=device)  # effectively -inf

        for _ in range(self.n_recursions):
            fused = torch.cat([x, y, z], dim=-1)   # [B, 3D]
            z_next = torch.tanh(self.z_proj(fused))  # [B, D]

            # Core transformation (attention+FFN or FFN-only)
            z_next = self.core(z_next)  # [B, D]

            # Halting logit for this step
            step_halt_logit = self.halt_head(self.final_ln(z_next))  # [B, 1]
            halt_logits = torch.maximum(halt_logits, step_halt_logit)

            # Residual update (small step)
            z = z + 0.1 * z_next

        logits = self.classifier(self.final_ln(z))   # [B, vocab_size]
        return logits, halt_logits.squeeze(-1), z
