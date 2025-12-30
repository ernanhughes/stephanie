# stephanie/model/vpm_vit.py
from __future__ import annotations

import torch
import torch.nn as nn


# ---------- utils ----------
def _build_2d_sincos_pos_embed(h: int, w: int, d: int, cls_token: bool = True, device=None):
    """2D sin-cos positional embedding (ViT-style), shape (1, 1+N, D) if cls_token."""
    assert d % 4 == 0, "D must be divisible by 4 for 2D sincos."
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = torch.arange(d // 4, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (d // 4)))
    yy = yy.reshape(-1, 1).float()
    xx = xx.reshape(-1, 1).float()
    out = torch.cat([
        torch.sin(yy * omega), torch.cos(yy * omega),
        torch.sin(xx * omega), torch.cos(xx * omega)
    ], dim=1)  # (N, D)
    out = out.unsqueeze(0)  # (1, N, D)
    if cls_token:
        out = torch.cat([torch.zeros(1, 1, d), out], dim=1)
    return out.to(device=device)

# ---------- modules ----------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, d_model: int = 384, patch: int = 8):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,N,D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x  # (B, N, D)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6, mlp_ratio: float = 4.0, p: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x

class VPMViT(nn.Module):
    """
    VPM-native ViT with optional MPM (masked patch token reconstruction).
    - Input: (B,C,H,W)
    - Output: {'reg': (B,T), 'cls': (B,K), 'mpm_rec': (M,D)} where M = total masked tokens across batch
    """
    def __init__(
        self,
        in_ch: int,
        d_model: int = 384,
        depth: int = 6,
        n_heads: int = 6,
        patch: int = 8,
        num_reg_targets: int = 5,
        num_risk_classes: int | None = 3,
        mlp_ratio: float = 4.0,
        p: float = 0.1,
        use_mpm: bool = True,
    ):
        super().__init__()
        self.patch = patch
        self.use_mpm = use_mpm
        self.patch_embed = PatchEmbed(in_ch, d_model, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed_cache: dict[tuple[int,int], torch.Tensor] = {}

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, mlp_ratio, p) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

        self.head_reg = nn.Linear(d_model, num_reg_targets) if num_reg_targets > 0 else None
        self.head_cls = nn.Linear(d_model, num_risk_classes) if num_risk_classes is not None else None
        self.mpm_head = nn.Linear(d_model, d_model) if use_mpm else None

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _pos_embed(self, H: int, W: int, D: int, device) -> torch.Tensor:
        # cache by (H', W') where H' = H//p, W' = W//p
        h = H // self.patch
        w = W // self.patch
        key = (h, w)
        pe = self.pos_embed_cache.get(key)
        if pe is None or pe.device != device or pe.shape[-1] != D:
            pe = _build_2d_sincos_pos_embed(h, w, D, cls_token=True, device=device)  # (1,1+N,D)
            self.pos_embed_cache[key] = pe
        return pe

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        patches = self.patch_embed(x)                 # (B, N, D)
        N, D = patches.shape[1], patches.shape[2]

        # tokens + pos
        cls = self.cls_token.expand(B, -1, -1)       # (B,1,D)
        xseq = torch.cat([cls, patches], dim=1)      # (B,1+N,D)
        xseq = xseq + self._pos_embed(H, W, D, x.device)

        for blk in self.blocks:
            xseq = blk(xseq)
        xseq = self.norm(xseq)

        cls_tok = xseq[:, 0]                         # (B,D)
        patch_tok = xseq[:, 1:]                      # (B,N,D)

        out: dict[str, torch.Tensor] = {}
        if self.head_reg is not None:
            out["reg"] = self.head_reg(cls_tok)
        if self.head_cls is not None:
            out["cls"] = self.head_cls(cls_tok)

        if self.use_mpm and mask is not None:
            # mask shape must be (B,N) boolean
            assert mask.dim() == 2 and mask.shape == (B, N), f"mask must be (B,N); got {tuple(mask.shape)}"
            rec = self.mpm_head(patch_tok)           # (B,N,D)
            out["mpm_rec"] = rec[mask]               # (M,D)
        return out

def vpm_vit_small(in_ch: int, targets: int = 5, classes: int = 3) -> VPMViT:
    return VPMViT(in_ch=in_ch, d_model=384, depth=6, n_heads=6, patch=8,
                  num_reg_targets=targets, num_risk_classes=classes, use_mpm=True)
