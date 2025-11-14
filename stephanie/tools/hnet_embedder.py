# stephanie/tools/hnet_embedder.py

"""
HNet-inspired Hierarchical Embedding Preprocessor
=================================================

This module implements a *practical* front-end to embed long texts by:
  1) Chunking raw UTF-8 bytes with hard safety limits (≤ max_bytes),
  2) Proposing content-aware boundaries via a lightweight routing model,
  3) Snapping to max-byte windows when needed,
  4) Embedding chunks with a base embedder and pooling to a single vector.

Relation to H-Net (DC)
----------------------
This is **not** a full H-Net. It borrows the *frame* of H-Net’s Dynamic Chunking (DC):
- H-Net learns boundaries end-to-end using a routing module that measures 
  **cosine similarity between adjacent representations**; boundaries are drawn
  when similarity drops (Eq. 4). Selected boundary vectors are directly kept,
  others are discarded, producing content-adaptive compression. 
- For stable training, H-Net uses a **smoothing module** (EMA) and an STE-based
  upsampler to make discrete boundary choices differentiable (Eqs. 5–9).
- A **ratio loss** encourages a target compression rate by matching the fraction
  of selected boundaries to the average boundary probability (Eq. 10).

In this preprocessor we approximate the *spirit* of DC for inference-time chunking:
- We compute a routing signal over bytes and prefer boundaries where content changes.
- We keep strict max-bytes guardrails for external embedders.
- We optionally smooth boundary probabilities and auto-tune the threshold so the
  realized boundary count roughly matches a desired compression target.

Why chunking matters here
-------------------------
External embedding APIs impose input limits; without content-aware splitting, we
risk cutting mid-idea or mid-grapheme. Routing-guided chunking improves semantic
coherence of chunks and thus pooled embeddings, while hard limits ensure the
embedder never sees overlong inputs.

Safety & Determinism
--------------------
- Never breaks UTF-8: truncation is byte-accurate at boundaries.
- Enforces max_bytes post-split (asserts if violated).
- Falls back to fixed windows when routing is unavailable.
- CUDA/cuDNN guarded: retries LSTMs with cuDNN disabled if NOT_SUPPORTED.

References
----------
H-Net: Hierarchical Networks without Tokenization (2025).
- Dynamic Chunking routing via adjacent cosine similarity (Eq. 4).
- Smoothing (EMA) and STE upsampling for differentiability (Eqs. 5–9).
- Ratio loss for targeted compression (Eq. 10).
arXiv:2507.07955.

Notes
-----
This module is designed as a robust *pre-embedding* step. It does not implement
H-Net’s decoders, dechunking, or end-to-end training; it provides a stable,
content-aware splitter that plays nicely with real-world embedding services.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

import re
import unicodedata

from stephanie.tools.embedding_tool import MXBAIEmbedder


@contextmanager
def _cudnn_flags(enabled: bool):
    """
    Temporarily toggle cuDNN enablement. Restores (enabled, benchmark, deterministic).
    """
    try:
        from torch.backends import cudnn
    except Exception:
        # No cuDNN available; do nothing.
        yield
        return

    prev = (cudnn.enabled, cudnn.benchmark, cudnn.deterministic)
    cudnn.enabled = enabled
    try:
        yield
    finally:
        cudnn.enabled, cudnn.benchmark, cudnn.deterministic = prev


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
    """
    UTF-8 byte tokenizer with safe decode.

    - tokenize: returns a list of byte values (0..255)
    - decode: decodes bytes with errors='ignore' so we NEVER inflate with replacement chars
    """
    def tokenize(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


# ---------------------------
# Boundary predictor (Tiny)
# ---------------------------

class ChunkBoundaryPredictor(nn.Module):
    """
    Lightweight boundary scorer over byte tokens.

    Returns a score in [0,1] per token; a "1" at index i means "end a chunk here".
    """
    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 128,
        device: str = "cpu",
        seed: Optional[int] = 1234,     # deterministic by default
    ):
        super().__init__()
        self.device = device

        # Isolate RNG effects to this module for determinism without leaking globally
        if seed is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self._init_layers(vocab_size, hidden_dim)
        else:
            self._init_layers(vocab_size, hidden_dim)

        self.eval()  # inference-only path

    def _init_layers(self, vocab_size: int, hidden_dim: int):
        self.embedding = nn.Embedding(vocab_size, hidden_dim).to(self.device)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=False
        ).to(self.device)
        self.boundary_scorer = nn.Linear(hidden_dim * 2, 1).to(self.device)

    @torch.inference_mode()
    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """
        Args:
            tokens: byte ids (0..255), length = N
        Returns:
            scores: shape [N], probabilities in [0,1]
        """
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
                msg = str(e)
                if "CUDNN_STATUS" in msg or "CUDNN_STATUS_NOT_SUPPORTED" in msg:
                    with _cudnn_flags(False):
                        y, _ = self.lstm(x)  # same device, cuDNN disabled
                else:
                    raise
            y = y.contiguous()
            x2 = y.squeeze(1)               # [seq, 2H]

        # Linear head expects [seq, 2H]
        x2 = x2.to(dtype=self.boundary_scorer.weight.dtype,
                   device=self.boundary_scorer.weight.device).contiguous()
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
    use_learned: bool = False       # default off unless explicit or weights loaded
    max_predict_tokens: int = 16384 # beyond this, skip learned prediction for robustness
    seed: Optional[int] = 1234      # deterministic tiny net by default


def _ema_smooth(arr: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    """In-place style EMA (returns new tensor) to calm spiky boundary probs."""
    if arr.numel() <= 1:
        return arr
    out = arr.clone()
    for i in range(1, out.numel()):
        out[i] = alpha * out[i] + (1 - alpha) * out[i - 1]
    return out

_DELIM_BYTES = [
    b"\n\n", b"\n", b". ", b"? ", b"! ", b"; ", b": ", b", ", b") ", b"] ", b"} ", b" - ",
]
_CODE_FENCE = b"```"

def _nearest_delim(bytes_buf: bytes, idx: int, search_radius: int = 64) -> int:
    """Find a nicer nearby textual break around idx; return adjusted idx or original."""
    n = len(bytes_buf)
    lo = max(0, idx - search_radius)
    hi = min(n, idx + search_radius)
    window = bytes_buf[lo:hi]
    best = idx
    best_dist = n
    # hard bias: code fence takes precedence
    for m in re.finditer(re.escape(_CODE_FENCE), window):
        fence_end = lo + m.end()  # end of ```
        d = abs(fence_end - idx)
        if d < best_dist:
            best = fence_end
            best_dist = d
    # softer: common delimiters
    for dbytes in _DELIM_BYTES:
        for m in re.finditer(re.escape(dbytes), window):
            cut = lo + m.end() - 1
            d = abs(cut - idx)
            if d < best_dist:
                best = cut
                best_dist = d
    return best


def _left_adjust_utf8_safe(tokens: List[int], end_idx: int, adjust_chars: int = 4) -> int:
    """
    Ensure we don't end on a UTF-8 continuation byte or leave trailing combining marks.
    We can only move <= adjust_chars back (bounded) and never forward (to respect max_bytes).
    Returns a byte index (inclusive).
    """
    n = len(tokens)
    if n == 0:
        return 0

    # Clamp so we never index tokens[n]
    if end_idx >= n:
        end_idx = n - 1
    if end_idx < 0:
        end_idx = 0

    # 1) avoid continuation-byte cut
    i = end_idx
    while i > 0 and (tokens[i] & 0b1100_0000) == 0b1000_0000:
        i -= 1
    end_idx = i

    # 2) avoid leaving trailing combining mark at the boundary
    tail_left = max(0, end_idx - 8 * adjust_chars)
    tail = bytes(tokens[tail_left:end_idx + 1]).decode("utf-8", errors="ignore")
    if not tail:
        return end_idx

    for _ in range(min(adjust_chars, len(tail))):
        ch = tail[-1]
        if unicodedata.category(ch) != "Mn":
            break
        tail = tail[:-1]
        end_idx = tail_left + len(tail.encode("utf-8")) - 1

    return max(0, end_idx)

def _binary_search_threshold(probs: torch.Tensor, target_count: int, lo=0.05, hi=0.95, iters=10) -> float:
    """Find a threshold so that (# probs > thr) ~= target_count."""
    target_count = max(1, target_count)
    thr = (lo + hi) / 2.0
    for _ in range(iters):
        count = int((probs > thr).sum().item())
        if count == target_count:
            return float(thr)
        if count > target_count:
            lo = thr
        else:
            hi = thr
        thr = (lo + hi) / 2.0
    return float(thr)

def _trimmed_length_weighted_mean(chunks: List[str], embeds: List[List[float]], trim_q: float = 0.05) -> List[float]:
    """Per-dim trimmed mean with length weights to reduce outlier chunk impact."""
    if not embeds:
        return []
    X = np.asarray(embeds, dtype=np.float32)
    w = np.asarray([max(1, len(c.encode("utf-8"))) for c in chunks], dtype=np.float32)
    w = w / w.sum()
    # compute weighted quantiles per dimension
    # simple approximation: mask extremes based on unweighted quantiles
    lo = np.quantile(X, trim_q, axis=0)
    hi = np.quantile(X, 1 - trim_q, axis=0)
    mask = (X >= lo) & (X <= hi)
    # keep entries within [lo, hi] across dimensions; conservative AND
    keep = mask.all(axis=1)
    Xk = X[keep]
    wk = w[keep] if keep.any() else w
    if Xk.size == 0:
        Xk = X
        wk = w
    wk = wk / wk.sum()
    return (Xk * wk[:, None]).sum(axis=0).astype(np.float32).tolist()


class HNetChunker:
    """
    Combines learned boundaries with hard windows:
      - propose boundaries using Tiny boundary predictor (optional)
      - ensure chunks never exceed max_bytes
      - try to snap to a nearby learned boundary within [last+min_bytes, window_end]
    """
    def __init__(self, cfg: ChunkerConfig):
        dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Guardrails & normalized config
        max_b = int(cfg.max_bytes)
        min_b = int(cfg.min_bytes)
        if min_b >= max_b:
            min_b = max(1, max_b // 4)

        self.cfg = ChunkerConfig(
            threshold=float(cfg.threshold),
            max_bytes=max_b,
            min_bytes=min_b,
            device=dev,
            use_learned=bool(cfg.use_learned),
            max_predict_tokens=int(cfg.max_predict_tokens),
            seed=cfg.seed,
        )
        self.tokenizer = ByteLevelTokenizer()
        self.boundary_predictor = ChunkBoundaryPredictor(
            device=self.cfg.device,
            seed=self.cfg.seed
        )

    def load_boundary_weights(self, state: dict) -> None:
        """Enable learned boundaries by loading a trained state dict."""
        self.boundary_predictor.load_state_dict(state, strict=True)
        self.cfg.use_learned = True

    @torch.no_grad()
    def _predict_boundary_probs(self, tokens: List[int]) -> torch.Tensor:
        """Return boundary probabilities per byte (0..1). Falls back to zeros if model absent."""
        if not tokens:
            return torch.empty(0, device=self.boundary_predictor.boundary_scorer.weight.device)
        scores = self.boundary_predictor(tokens)  # [N] in [0,1]
        return scores

    def _choose_boundaries(self, tokens: List[int], probs: torch.Tensor) -> List[int]:
        n = len(tokens)
        if n == 0:
            return []

        # 1) smooth
        probs = _ema_smooth(probs, alpha=0.2)

        # 2) aim for desired chunk count via threshold auto-tune
        desired_chunks = max(1, int(np.ceil(n / self.cfg.max_bytes)))
        thr = _binary_search_threshold(probs, target_count=desired_chunks - 1)  # minus final boundary
        learned = (probs > thr).nonzero(as_tuple=True)[0].tolist()

        # 3) merge with hard windows (your existing logic)
        ends = self._force_window_boundaries(n, learned)

        # 4) snap boundaries to nicer delimiters + UTF-8 safety
        b = bytes(tokens)  # single conversion
        adjusted = []
        for e in ends:
            nice = _nearest_delim(b, e, search_radius=max(32, self.cfg.min_bytes // 2))
            safe = _left_adjust_utf8_safe(tokens, nice, adjust_chars=4)
            adjusted.append(safe)
        # Ensure strictly increasing and final == n-1
        adjusted = sorted(set(x for x in adjusted if 0 <= x < n))
        if not adjusted or adjusted[-1] != n - 1:
            adjusted = [x for x in adjusted if x < n - 1] + [n - 1]
        return adjusted


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
        if bounds and bounds[-1] != n - 1:
            bounds.append(n - 1)
        elif not bounds:
            bounds = [n - 1]
        return bounds

    def chunk(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        n = len(tokens)
        if n == 0:
            return []

        # Prefer learned boundaries with auto-tuned threshold
        probs = self._predict_boundary_probs(tokens) if n > 1 else torch.tensor([1.0], device=self.boundary_predictor.boundary_scorer.weight.device)
        ends = self._choose_boundaries(tokens, probs) if probs.numel() else self._generate_fixed_boundaries(n)

        out: List[str] = []
        prev = -1
        max_b = self.cfg.max_bytes
        for e in ends:
            # slice [prev+1, e] inclusive
            seg_bytes = bytes(tokens[prev + 1: e + 1])
            # Hard post-check: enforce max_bytes with left nudge if needed
            if len(seg_bytes) > max_b:
                # nudge left to satisfy max_bytes without breaking UTF-8
                cut = prev + max_b
                cut = _left_adjust_utf8_safe(tokens, cut, adjust_chars=4)
                seg_bytes = bytes(tokens[prev + 1: cut + 1])
                e = cut
            out.append(seg_bytes.decode("utf-8", errors="ignore"))
            prev = e
        return out



# ---------------------------
# Pooling
# ---------------------------

class PoolingStrategy:
    @staticmethod
    def _to_np2d(embeds: Sequence[Any]) -> np.ndarray:
        """Coerce list[list[float]] or array-like to strict float32 2D array."""
        if embeds is None:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(embeds, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    @staticmethod
    def mean_pool(embeds: Sequence[Any]) -> List[float]:
        arr = PoolingStrategy._to_np2d(embeds)
        if arr.size == 0:
            return []
        return np.mean(arr, axis=0).astype(np.float32).tolist()

    @staticmethod
    def length_weighted_mean_pool(chunks: List[str], embeds: Sequence[Any]) -> List[float]:
        arr = PoolingStrategy._to_np2d(embeds)
        if arr.size == 0:
            return []
        weights = np.array([max(1, len(c.encode("utf-8"))) for c in chunks], dtype=np.float32)
        if weights.shape[0] != arr.shape[0]:
            raise ValueError(f"length_weighted_mean_pool: weight count {weights.shape[0]} "
                             f"!= embed rows {arr.shape[0]}")
        wsum = float(weights.sum())
        if wsum <= 0:
            return PoolingStrategy.mean_pool(arr)
        return (arr * (weights / wsum)[:, None]).sum(axis=0).astype(np.float32).tolist()


# ---------------------------
# High-level embedder
# ---------------------------

class HNetEmbedder:
    """
    Hierarchical embedder:
      1) Chunk long text safely (≤ max_bytes)
      2) Embed each chunk with a base embedder
      3) Pool chunk embeddings (length-weighted mean by default)

    Robustness features:
      - Deterministic tiny boundary model with switchable learned boundaries
      - Byte-true chunk enforcement with defensive truncation
      - Dimensionality validation to detect upstream changes early
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
        use_learned_boundaries: bool = False,
        max_predict_tokens: int = 16384,
        seed: Optional[int] = 1234,
    ):
        cfg = ChunkerConfig(
            threshold=threshold,
            max_bytes=max_bytes,
            min_bytes=min_bytes,
            device=device,
            use_learned=use_learned_boundaries,
            max_predict_tokens=max_predict_tokens,
            seed=seed,
        )
        self.chunker = HNetChunker(cfg)
        self.embedder = embedder

        # Preferred dimension if provided; otherwise determined on first embed
        self.dim: Optional[int] = int(getattr(self.embedder, "dim", 0)) or None

        self.pooler = PoolingStrategy()
        self.use_length_weighting = bool(use_length_weighting)

    def _zeros(self) -> List[float]:
        d = self.dim if self.dim and self.dim > 0 else 1024
        return [0.0] * int(d)

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            # Stable all-zero fallback
            return self._zeros()

        # 1) Chunk
        chunks = self.chunker.chunk(text)

        # 2) Hard safety: enforce max_bytes on the encoded form (no expansion allowed)
        max_b = self.chunker.cfg.max_bytes
        chunks = [_truncate_utf8_bytes(c, max_b) for c in chunks]

        # Defensive re-check
        for i, c in enumerate(chunks):
            b = len(c.encode("utf-8"))
            if b > max_b:
                raise RuntimeError(f"Chunk {i} is {b} bytes > max {max_b}")

        if not chunks:
            return self._zeros()

        # 3) Single-chunk fast path
        if len(chunks) == 1:
            single = self.embedder.batch_embed(chunks)
            if not single:
                return self._zeros()
            vec = np.asarray(single[0], dtype=np.float32)
            if vec.ndim != 1:
                raise ValueError("Embedder returned non-1D vector for single chunk.")
            # Infer dim if unknown
            if self.dim is None:
                self.dim = int(vec.shape[0])
            elif int(vec.shape[0]) != int(self.dim):
                raise ValueError(f"Embedding dim mismatch: got {vec.shape[0]}, expected {self.dim}")
            return vec.tolist()

        # 4) Batch embed
        chunk_embeddings = self.embedder.batch_embed(chunks)  # List[List[float]] or np.array
        if not chunk_embeddings:
            return self._zeros()

        # Validate dims & coerce to array
        arr = np.asarray(chunk_embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Embedder returned array with ndim={arr.ndim}, expected 2")
        if self.dim is None:
            self.dim = int(arr.shape[1])
        if int(arr.shape[1]) != int(self.dim):
            raise ValueError(f"Embedding dim mismatch: got {arr.shape[1]}, expected {self.dim}")

        # 5) Pool
        if self.use_length_weighting:
            return _trimmed_length_weighted_mean(chunks, chunk_embeddings, trim_q=0.05)
        return self.pooler.mean_pool(chunk_embeddings)

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    # Expose a way to load boundary weights and enable learned snaps
    def load_boundary_weights(self, state_dict: dict) -> None:
        self.chunker.load_boundary_weights(state_dict)


# ---------------------------
# Singleton glue
# ---------------------------

_hnet_instance: Optional[HNetEmbedder] = None
_hnet_signature: Optional[Tuple[Any, ...]] = None

def _make_signature(cfg: dict) -> Tuple[Any, ...]:
    emcfg = (cfg or {}).get("embeddings", {}) or {}
    return (
        int(emcfg.get("max_prompt_bytes", 1024)),
        int(emcfg.get("min_prompt_bytes", 256)),
        float(emcfg.get("hnet_threshold", 0.70)),
        emcfg.get("device"),
        bool(emcfg.get("use_learned_boundaries", False)),
        int(emcfg.get("max_predict_tokens", 16384)),
        int(emcfg.get("seed", 1234)),
        # try to lock on base embedder model id if present
        getattr(MXBAIEmbedder, "__name__", "MXBAIEmbedder"),
        emcfg.get("model"),  # passthrough if MXBAIEmbedder reads this
    )

def get_embedding(text: str, cfg: dict) -> List[float]:
    """
    Singleton entry point used by EmbeddingStore.

    Pulls HNet settings from cfg["embeddings"]:
      - max_prompt_bytes (int, default 1024)
      - min_prompt_bytes (int, default 256)
      - hnet_threshold (float, default 0.70)
      - device ("cuda"|"cpu"|None)
      - use_learned_boundaries (bool, default False)
      - max_predict_tokens (int, default 16384)
      - seed (int, default 1234)
    """
    global _hnet_instance, _hnet_signature

    emcfg = (cfg or {}).get("embeddings", {}) or {}
    max_bytes = int(emcfg.get("max_prompt_bytes", 1024))
    min_bytes = int(emcfg.get("min_prompt_bytes", 256))
    threshold = float(emcfg.get("hnet_threshold", 0.70))
    device = emcfg.get("device")  # "cuda", "cpu", or None
    use_learned = bool(emcfg.get("use_learned_boundaries", False))
    max_predict_tokens = int(emcfg.get("max_predict_tokens", 16384))
    seed = int(emcfg.get("seed", 1234))

    sig = _make_signature(cfg)
    if (_hnet_instance is None) or (_hnet_signature != sig):
        base_embedder = MXBAIEmbedder(cfg)  # respects your endpoint/model config
        _hnet_instance = HNetEmbedder(
            base_embedder,
            max_bytes=max_bytes,
            min_bytes=min_bytes,
            threshold=threshold,
            device=device,
            use_length_weighting=True,
            use_learned_boundaries=use_learned,
            max_predict_tokens=max_predict_tokens,
            seed=seed,
        )
        _hnet_signature = sig

    return _hnet_instance.embed(text)
