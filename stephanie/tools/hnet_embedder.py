# stephanie/tools/hnet_embedder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from stephanie.tools.embedding_tool import MXBAIEmbedder


# ---------------------------
# Tokenizer (byte-level)
# ---------------------------

class ByteLevelTokenizer:
    """UTF-8 byte tokenizer with safe decode (replacement char for bad splits)."""

    def tokenize(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        # Safe even when we cut through a multi-byte codepoint
        return bytes(tokens).decode("utf-8", errors="replace")


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

    @torch.no_grad()
    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """
        Args:
            tokens: byte ids (0..255), length = N
        Returns:
            scores: shape [N], probabilities in [0,1]
        """
        if not isinstance(tokens, torch.Tensor):
            tok = torch.tensor(tokens, dtype=torch.long, device=self.device)
        else:
            tok = tokens.detach().clone().long().to(self.device)

        x = self.embedding(tok).float()  # [N, H]

        # For N==1, emulate biLSTM output width without calling LSTM
        if x.size(0) == 1:
            x2 = torch.cat([x, torch.zeros_like(x)], dim=-1)  # [1, 2H]
        else:
            # [seq, 1, H]; disable cuDNN to avoid RNN oddities on tiny or dynamic seqs
            x = x.unsqueeze(1).contiguous()
            x, _ = self.lstm(x)  # [seq, 1, 2H]
            x2 = x.squeeze(1)  # [seq, 2H]

        scores = self.boundary_scorer(x2)  # [N, 1]
        return scores.sigmoid().flatten()   # [N]


# ---------------------------
# Chunker with hard limits
# ---------------------------

@dataclass
class ChunkerConfig:
    threshold: float = 0.70         # boundary score threshold
    max_bytes: int = 1024           # hard upper bound per chunk (bytes)
    min_bytes: int = 256            # soft lower bound for snapping
    device: Optional[str] = None    # "cuda" | "cpu" | None -> auto


class StephanieHNetChunker:
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

class StephanieHNetEmbedder:
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
        self.chunker = StephanieHNetChunker(cfg)
        self.embedder = embedder
        self.dim = int(getattr(self.embedder, "dim", 1024))
        self.pooler = PoolingStrategy()
        self.use_length_weighting = bool(use_length_weighting)

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            # Stable all-zero fallback
            return [0.0] * self.dim

        chunks = self.chunker.chunk(text)
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

_hnet_instance: Optional[StephanieHNetEmbedder] = None

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
        _hnet_instance = StephanieHNetEmbedder(
            base_embedder,
            max_bytes=max_bytes,
            min_bytes=min_bytes,
            threshold=threshold,
            device=device,
            use_length_weighting=True,
        )

    return _hnet_instance.embed(text)
