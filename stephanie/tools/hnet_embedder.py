# stephanie/tools/hnet_embedder.py
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from stephanie.tools.embedding_tool import MXBAIEmbedder


@contextmanager
def _cudnn_flags(enabled: bool):
    """Scoped cuDNN enable/disable that won’t leak."""
    try:
        from torch.backends import cudnn
        prev = (cudnn.enabled, cudnn.benchmark, cudnn.deterministic)
        cudnn.enabled = enabled
        # we keep benchmark/deterministic unchanged
        yield
        cudnn.enabled, cudnn.benchmark, cudnn.deterministic = prev
    except Exception:
        # Fallback: do nothing if flags API isn’t available
        yield


# ---------------------------
# Tokenizer (byte-level)
# ---------------------------

def _truncate_utf8_bytes(s: str, max_bytes: int) -> str:
    """Ensure s encodes to at most max_bytes in UTF-8 by truncating at a byte boundary."""
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", errors="ignore")

class ByteLevelTokenizer:
    """UTF-8 byte tokenizer with safe decode (replacement char for bad splits)."""

    def tokenize(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        # Use 'ignore' so decoding never inserts multi-byte replacement chars.
        return bytes(tokens).decode("utf-8", errors="ignore")


# ---------------------------
# Boundary predictor (Tiny)
# ---------------------------
class ChunkBoundaryPredictor(nn.Module):
    """
    Lightweight boundary scorer over byte tokens.

    Returns a score in [0,1] per token; a "1" at index i means "end a chunk here".
    """
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, hidden_dim).to(self.device)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=False).to(self.device)
        self.boundary_scorer = nn.Linear(hidden_dim * 2, 1).to(self.device)
        self.eval()  # we only ever infer here

    @torch.no_grad()
    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """
        Args:
            tokens: byte ids (0..255), length = N
        Returns:
            scores: shape [N], probabilities in [0,1]
        """
        # --- Guard 0/1 length early
        if not isinstance(tokens, torch.Tensor):
            tok = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        else:
            tok = tokens.detach().to(self.device).long()

        N = tok.numel()
        if N == 0:
            return torch.empty(0, device=self.device)

        x = self.embedding(tok).float()  # [N, H]

        if N == 1:
            # emulate biLSTM width without invoking cuDNN kernels
            x2 = torch.cat([x, torch.zeros_like(x)], dim=-1)  # [1, 2H]
        else:
            # Prepare for LSTM: [seq, 1, H], contiguous
            x = x.unsqueeze(1).contiguous()

            # Help cuDNN choose a fast kernel in the common case
            try:
                if torch.backends.cudnn.enabled:
                    self.lstm.flatten_parameters()
            except Exception:
                pass

            # Try cuDNN first; if it barfs with NOT_SUPPORTED, retry with cuDNN off.
            try:
                y, _ = self.lstm(x)         # [seq, 1, 2H]
            except RuntimeError as e:
                if "CUDNN_STATUS_NOT_SUPPORTED" in str(e):
                    with _cudnn_flags(False):
                        y, _ = self.lstm(x)  # same device, cuDNN disabled
                else:
                    raise
            y = y.contiguous()
            x2 = y.squeeze(1)               # [seq, 2H]

        # Linear head expects [seq, 2H]
        x2 = x2.contiguous()
        scores = self.boundary_scorer(x2)    # [N, 1]
        return scores.sigmoid().flatten()    # [N]


# ---------------------------
# Chunker with hard limits
# ---------------------------

@dataclass
class ChunkerConfig:
    threshold: float = 0.70         # boundary score threshold
    max_bytes: int = 1024           # hard upper bound per chunk (bytes)
    min_bytes: int = 256            # soft lower bound for snapping
    device: Optional[str] = None    # "cuda" | "cpu" | None -> auto


class HNetChunker:
    """
    Combines learned boundaries with hard windows:
      - propose boundaries using Tiny boundary predictor
      - ensure chunks never exceed max_bytes
      - try to snap to a nearby learned boundary within [last+min_bytes, window_end]
    """
    def __init__(self, cfg: ChunkerConfig):
        dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = ChunkerConfig(
            threshold=float(cfg.threshold),
            max_bytes=int(cfg.max_bytes),
            min_bytes=int(cfg.min_bytes),
            device=dev,
        )
        self.tokenizer = ByteLevelTokenizer()
        self.boundary_predictor = ChunkBoundaryPredictor(device=self.cfg.device)

        # Guardrails
        if self.cfg.min_bytes >= self.cfg.max_bytes:
            # keep a safe gap
            self.cfg.min_bytes = max(1, self.cfg.max_bytes // 4)

    @torch.no_grad()
    def _predict_boundaries(self, tokens: List[int]) -> List[int]:
        """Return candidate boundary indices where score > threshold."""
        if not tokens:
            return []
        scores = self.boundary_predictor(tokens)  # [N]
        cand = (scores > self.cfg.threshold).nonzero(as_tuple=True)[0].tolist()
        return cand

    def _generate_fixed_boundaries(self, n: int) -> List[int]:
        """
        Fallback: fixed windows of size max_bytes.
        Emit end indices: (max_bytes-1), (2*max_bytes-1), ..., and always n-1.
        """
        step = self.cfg.max_bytes
        if n <= step:
            return [n - 1]

        # 1023, 2047, 3071, ... (< n-1)
        ends = list(range(step - 1, n - 1, step))

        # ensure final boundary lands exactly at end
        if not ends or ends[-1] != n - 1:
            ends.append(n - 1)
        return ends

    def _force_window_boundaries(self, n: int, learned: List[int]) -> List[int]:
        """
        Merge learned boundaries with hard windows.

        Strategy:
          - Walk forward from last boundary, open a window of size max_bytes.
          - Prefer the latest learned boundary inside [last+min_bytes, window_end].
          - If none found, use window_end.
          - Always finalize with n-1 as the last boundary.
        """
        lb = sorted(set(b for b in learned if 0 <= b < n))
        bounds: List[int] = []
        last = -1

        while last < n - 1:
            window_end = min(last + self.cfg.max_bytes, n - 1)
            # learned snap candidates after we’ve satisfied min_bytes
            lo = last + self.cfg.min_bytes
            cand = [b for b in lb if lo <= b <= window_end]
            snap = max(cand) if cand else window_end
            if snap <= last:   # safety, shouldn’t happen
                snap = min(last + self.cfg.max_bytes, n - 1)
            bounds.append(snap)
            last = snap
            if snap == n - 1:
                break

        # Always ensure the stream ends exactly at n-1 (dedup safe)
        if bounds[-1] != n - 1:
            bounds[-1] = n - 1
        return bounds

    def chunk(self, text: str) -> List[str]:
        """
        Split into chunks (each ≤ max_bytes), trying to honor learned semantic boundaries.
        """
        tokens = self.tokenizer.tokenize(text)
        n = len(tokens)
        if n == 0:
            return []

        learned = self._predict_boundaries(tokens)
        if not learned:
            ends = self._generate_fixed_boundaries(n)
        else:
            ends = self._force_window_boundaries(n, learned)

        out: List[str] = []
        prev = -1
        for b in ends:
            # slice [prev+1, b] inclusive
            piece = self.tokenizer.decode(tokens[prev + 1: b + 1])
            out.append(piece)
            prev = b
        return out


# ---------------------------
# Pooling
# ---------------------------

class PoolingStrategy:
    @staticmethod
    def mean_pool(embeds: List[List[float]]) -> List[float]:
        return np.mean(embeds, axis=0).tolist() if embeds else []

    @staticmethod
    def length_weighted_mean_pool(chunks: List[str], embeds: List[List[float]]) -> List[float]:
        if not embeds:
            return []
        weights = [max(1, len(c.encode("utf-8"))) for c in chunks]
        return np.average(embeds, weights=weights, axis=0).tolist()


# ---------------------------
# High-level embedder
# ---------------------------

class HNetEmbedder:
    """
    Hierarchical embedder:
      1) Chunk long text safely (≤ max_bytes)
      2) Embed each chunk with a base embedder
      3) Pool chunk embeddings (length-weighted mean)
    """
    def __init__(
        self,
        embedder: MXBAIEmbedder,
        *,
        max_bytes: int = 1024,
        min_bytes: int = 256,
        threshold: float = 0.70,
        device: Optional[str] = None,
        use_length_weighting: bool = True,
    ):
        cfg = ChunkerConfig(
            threshold=threshold,
            max_bytes=max_bytes,
            min_bytes=min_bytes,
            device=device,
        )
        self.chunker = HNetChunker(cfg)
        self.embedder = embedder
        self.dim = int(getattr(self.embedder, "dim", 1024))
        self.pooler = PoolingStrategy()
        self.use_length_weighting = bool(use_length_weighting)

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            # Stable all-zero fallback
            return [0.0] * self.dim

        chunks = self.chunker.chunk(text)

        # Hard safety: enforce max_bytes on the encoded form (no expansion allowed)
        max_b = self.chunker.cfg.max_bytes
        chunks = [_truncate_utf8_bytes(c, max_b) for c in chunks]


        # Debug guard (you can keep as log or assert during testing)
        for i, c in enumerate(chunks):
            b = len(c.encode("utf-8"))
            if b > self.chunker.cfg.max_bytes:
                raise RuntimeError(f"Chunk {i} is {b} bytes > max {self.chunker.cfg.max_bytes}")

        if not chunks:
            return [0.0] * self.dim

        chunk_embeddings = self.embedder.batch_embed(chunks)
        if self.use_length_weighting:
            return self.pooler.length_weighted_mean_pool(chunks, chunk_embeddings)
        return self.pooler.mean_pool(chunk_embeddings)

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------
# Singleton glue
# ---------------------------

_hnet_instance: Optional[HNetEmbedder] = None

def get_embedding(text: str, cfg: dict) -> List[float]:
    """
    Singleton entry point used by EmbeddingStore.

    Pulls HNet settings from cfg["embeddings"]:
      - max_prompt_bytes (int, default 1024)
      - min_prompt_bytes (int, default 256)
      - hnet_threshold (float, default 0.70)
    """
    global _hnet_instance

    emcfg = (cfg or {}).get("embeddings", {}) or {}
    max_bytes = int(emcfg.get("max_prompt_bytes", 1024))
    min_bytes = int(emcfg.get("min_prompt_bytes", 256))
    threshold = float(emcfg.get("hnet_threshold", 0.70))
    device = emcfg.get("device")  # "cuda", "cpu", or None

    if _hnet_instance is None:
        base_embedder = MXBAIEmbedder(cfg)  # respects your endpoint/model config
        _hnet_instance = HNetEmbedder(
            base_embedder,
            max_bytes=max_bytes,
            min_bytes=min_bytes,
            threshold=threshold,
            device=device,
            use_length_weighting=True,
        )

    return _hnet_instance.embed(text)
