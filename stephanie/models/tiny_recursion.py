# stephanie/models/tiny_recursion.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyBlock(nn.Module):
    """Basic nonlinear block with SwiGLU + residual + norm."""
    def __init__(self, d_in: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 2 * d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        h = F.silu(a) * b
        h = self.fc2(h)
        h = self.proj(h)
        h = self.dropout(h)
        return self.norm(h + x[..., :h.shape[-1]])


class TinyBlockAttn(nn.Module):
    """Attention-enabled variant of TinyBlock."""
    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_in)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_in,
            num_heads=n_heads,
            dropout=dropout,
            bias=attn_bias,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_in, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm_in(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        h = self.ffn(x)
        h = self.proj(h)
        return self.norm_out(h + x[..., :h.shape[-1]])


class TinyRecursionModel(nn.Module):
    """
    TinyRecursionModel
    ------------------
    • Optionally attention-enabled via `use_attention=True`
    • Uses recursive reasoning over (x, y, z) triplets
    • Predicts next-step reasoning logits and halting probability
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 2,
        n_recursions: int = 6,
        vocab_size: int = 1024,
        use_attention: bool = False,
        dropout: float = 0.0,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_recursions = n_recursions
        self.use_attention = use_attention

        BlockClass = TinyBlockAttn if use_attention else TinyBlock
        self.blocks = nn.ModuleList(
            [BlockClass(d_model * 3, d_model, dropout=dropout) for _ in range(n_layers)]
        )

        self.output_head = nn.Linear(d_model, vocab_size)
        self.halt_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """
        Args:
            x, y, z: (B, T, D)
        Returns:
            logits: (B, T, vocab_size)
            halt_p: (B, 1)
            z: final latent
        """
        for _ in range(self.n_recursions):
            inp = torch.cat([x, y, z], dim=-1)
            for blk in self.blocks:
                z = blk(inp)
            y = self.blocks[0](torch.cat([y, z], dim=-1))

        logits = self.output_head(y)
        halt_p = torch.sigmoid(self.halt_head(y.mean(dim=1)))
        return logits, halt_p, z
