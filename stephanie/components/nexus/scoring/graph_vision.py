#
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from stephanie.components.nexus.graph.graph_layout import \
    render_multi_layout_vpm

try:
    import torch  # optional; used only if a jit model is supplied
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@dataclass
class VisionConfig:
    """
    Lightweight knobs. Keep this aligned with the probe/report scripts.
    """
    img_size: int = 256
    cache_dir: Optional[str] = ".vision_cache"
    use_learned_head: bool = True  # if a model is present/valid
    # Symmetry
    symmetry_axis_blend: float = 0.5  # blend horizontal/vertical
    symmetry_epsilon: float = 1e-8
    # Bridge proxy (bottleneck concentration)
    grid_bins: int = 32        # histogram bins per axis for concentration calc
    bridge_alpha_col: float = 0.6  # column concentration weight
    bridge_alpha_row: float = 0.4  # row concentration weight
    bridge_epsilon: float = 1e-8
    # Spectral gap buckets
    gap_lo: float = 0.012
    gap_hi: float = 0.050
    # Learned head feature scaling
    feat_eps: float = 1e-6


class VisionScorer:
    """
    Dependency-light structural perception over VPMs.

    Public:
      - score_graph(G, cache_key=None) -> dict
      - score_from_vpm(vpms, metas) -> dict

    Returns (at least):
      {
        "vision_symmetry": float in [0,1],
        "vision_bridge_proxy": float in [0,1],
        "vision_spectral_gap_bucket": int in {0,1,2},
        "per_layout": [
            {
              "layout": "forceatlas2",
              "symmetry": ...,
              "bridge_proxy": ...,
              "gap_bucket": ...,
              "fallback": "spring" | None
            }, ...
        ],
      }
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", cfg: Optional[VisionConfig] = None):
        self.cfg = cfg or VisionConfig()
        self.device = device
        self.model = None
        self.model_ok = False

        if model_path and TORCH_AVAILABLE:
            p = Path(model_path)
            if p.exists():
                try:
                    # Expecting a TorchScript that maps feature vector -> refined scores (optional)
                    self.model = torch.jit.load(str(p), map_location=device)
                    self.model.eval()
                    self.model_ok = True
                except Exception:
                    self.model = None
                    self.model_ok = False

        if self.cfg.cache_dir:
            Path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------- Public API ----------------------

    def score_graph(self, G: nx.Graph, cache_key: Optional[str] = None) -> Dict[str, float]:
        """
        Render multi-layout VPMs, compute structural vision metrics, optionally use cache.
        """
        ck = cache_key or self._default_cache_key(G)
        if self.cfg.cache_dir:
            cached = self._try_cache_load(ck)
            if cached is not None:
                return cached

        # Render FA2+Spectral VPMs
        vpms, metas = render_multi_layout_vpm(
            G,
            layouts=("forceatlas2", "spectral"),
            config={
                "img_size": self.cfg.img_size,
                "cache_dir": self.cfg.cache_dir and str(Path(self.cfg.cache_dir) / "layout_cache")
            },
        )
        out = self.score_from_vpm(vpms, metas)

        if self.cfg.cache_dir:
            self._cache_save(ck, out)
        return out

    def score_from_vpm(self, vpms: List[np.ndarray], metas: List[Dict]) -> Dict[str, float]:
        """
        Compute metrics from already-rendered VPM tensors and layout metadata.

        vpms: list of [C,H,W] uint8 arrays (channels-first)
        metas: layout metadata list from render_multi_layout_vpm
        """
        per_layout: List[Dict[str, float]] = []
        sym_vals: List[float] = []
        bridge_vals: List[float] = []
        gap_buckets: List[int] = []

        for vpm, meta in zip(vpms, metas):
            # Expect channels-first [C,H,W]; ensure float in [0,1]
            vpmf = vpm.astype(np.float32) / 255.0
            node = self._safe_channel(vpmf, 0)  # node_density
            edge = self._safe_channel(vpmf, 1)  # edge_density
            # degree heat exists but not needed directly here

            sym = self._symmetry_score(node)
            brg = self._bridge_proxy(edge)
            gap_bucket = self._gap_bucket(meta.get("spectral_gap", 0.0))

            per_layout.append({
                "layout": str(meta.get("layout")),
                "symmetry": float(sym),
                "bridge_proxy": float(brg),
                "gap_bucket": int(gap_bucket),
                "fallback": meta.get("layout_fallback"),
            })

            sym_vals.append(sym)
            bridge_vals.append(brg)
            gap_buckets.append(gap_bucket)

        # Aggregate across layouts
        # - Symmetry: take max (the best, most revealing symmetry)
        # - Bridge: take max (if any layout reveals a concentrated bridge, flag it)
        # - Gap bucket: take min (pessimistic connectivity)
        agg_sym = float(np.max(sym_vals) if sym_vals else 0.0)
        agg_brg = float(np.max(bridge_vals) if bridge_vals else 0.0)
        agg_gap = int(np.min(gap_buckets) if gap_buckets else 0)

        # Optional learned refinement
        if self.model_ok and self.cfg.use_learned_head:
            try:
                feats = self._make_feature_vector(per_layout)
                with torch.no_grad():
                    t = torch.from_numpy(feats).float().unsqueeze(0)
                    if self.device:
                        t = t.to(self.device)
                    pred = self.model(t)  # expect either dict-like or tensor
                    if isinstance(pred, dict):
                        agg_sym = float(pred.get("symmetry", agg_sym))
                        agg_brg = float(pred.get("bridge_proxy", agg_brg))
                        agg_gap = int(pred.get("gap_bucket", agg_gap))
                    else:
                        # Assume tensor [B,3] -> sym, bridge, gap_bucket (logits)
                        p = pred.squeeze(0)
                        if p.numel() >= 3:
                            agg_sym = float(p[0].clamp(0, 1).item())
                            agg_brg = float(p[1].clamp(0, 1).item())
                            agg_gap = int(int(torch.argmax(p[2:].softmax(-1)).item()))
            except Exception:
                # Silently keep heuristic values
                pass

        return {
            "vision_symmetry": agg_sym,
            "vision_bridge_proxy": agg_brg,
            "vision_spectral_gap_bucket": agg_gap,
            "per_layout": per_layout,
        }

    # ---------------------- Core metrics ----------------------

    def _symmetry_score(self, img: np.ndarray) -> float:
        """
        Reflection coherence over horizontal & vertical axes.
        Returns in [0,1]. Higher = more symmetric mass distribution.
        """
        # Normalize channel
        x = img.astype(np.float32)
        x = (x - x.min()) / (x.max() - x.min() + self.cfg.symmetry_epsilon)
        # Horizontal reflection
        h_corr = self._reflection_corr(x, axis=1)  # left-right
        # Vertical reflection
        v_corr = self._reflection_corr(x, axis=0)  # top-bottom
        return float(self.cfg.symmetry_axis_blend * h_corr + (1 - self.cfg.symmetry_axis_blend) * v_corr)

    def _bridge_proxy(self, edge_ch: np.ndarray) -> float:
        """
        Measures how concentrated the edge mass is along narrow bands (bottleneck).
        Hybrid of column/row concentration indices with Herfindahl/Gini flavor.
        Returns in [0,1], where higher ~ stronger single-bridge signature.
        """
        x = edge_ch.astype(np.float32)
        if x.size == 0 or float(x.max()) < self.cfg.bridge_epsilon:
            return 0.0

        H, W = x.shape
        # Column-wise mass distribution
        col_mass = x.sum(axis=0)  # [W]
        row_mass = x.sum(axis=1)  # [H]

        # Downsample into fixed bins for stability
        def bin_reduce(v: np.ndarray, bins: int) -> np.ndarray:
            L = v.shape[0]
            if L <= bins:
                # pad
                out = np.zeros(bins, dtype=np.float32)
                out[:L] = v
                return out
            # average-pool
            k = L // bins
            trimmed = v[:k * bins].reshape(bins, k).mean(axis=1)
            return trimmed

        cb = bin_reduce(col_mass, self.cfg.grid_bins)
        rb = bin_reduce(row_mass, self.cfg.grid_bins)

        cb = cb / (cb.sum() + self.cfg.bridge_epsilon)
        rb = rb / (rb.sum() + self.cfg.bridge_epsilon)

        # Herfindahl index (sum of squares) — higher if mass is concentrated in a few bins
        h_col = float(np.sum(cb ** 2))
        h_row = float(np.sum(rb ** 2))

        # Normalize against uniform baseline (1/bins) -> map to [0,1]
        base = 1.0 / float(self.cfg.grid_bins)
        def norm_h(h):
            # h in [base,1]; map base→0, 1→1
            return float((h - base) / (1.0 - base + self.cfg.bridge_epsilon))

        c_idx = norm_h(h_col)
        r_idx = norm_h(h_row)

        # Weighted blend of column and row concentration
        score = self.cfg.bridge_alpha_col * c_idx + self.cfg.bridge_alpha_row * r_idx
        # Cap to [0,1]
        return float(max(0.0, min(1.0, score)))

    def _gap_bucket(self, gap: float) -> int:
        """
        0 = low (fragile), 1 = mid, 2 = high (robust)
        """
        if gap < self.cfg.gap_lo:
            return 0
        if gap < self.cfg.gap_hi:
            return 1
        return 2

    # ---------------------- Helpers ----------------------

    def _reflection_corr(self, x: np.ndarray, axis: int) -> float:
        """
        Correlate an image with its reflection along given axis.
        axis=1: left-right; axis=0: top-bottom
        """
        if axis == 1:
            xr = np.flip(x, axis=1)
        else:
            xr = np.flip(x, axis=0)

        # Zero-mean, unit variance
        a = x - x.mean()
        b = xr - xr.mean()
        denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()) + self.cfg.symmetry_epsilon)
        corr = float((a * b).sum() / denom)
        # Map correlation [-1,1] -> [0,1]
        return 0.5 * (corr + 1.0)

    def _safe_channel(self, vpmf: np.ndarray, idx: int) -> np.ndarray:
        if vpmf.ndim != 3:
            raise ValueError("VPM must be [C,H,W]")
        C, H, W = vpmf.shape
        if idx >= C:
            # repeat last
            return vpmf[-1]
        return vpmf[idx]

    def _default_cache_key(self, G: nx.Graph) -> str:
        # Simple structural signature
        n = G.number_of_nodes()
        m = G.number_of_edges()
        deg = sorted([d for _, d in G.degree()])[:32]
        payload = json.dumps({"n": n, "m": m, "deg": deg}, separators=(",", ":"))
        import hashlib
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _try_cache_load(self, key: str) -> Optional[Dict[str, float]]:
        try:
            p = Path(self.cfg.cache_dir) / f"{key}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return None
        return None

    def _cache_save(self, key: str, obj: Dict[str, float]) -> None:
        try:
            p = Path(self.cfg.cache_dir) / f"{key}.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _make_feature_vector(self, per_layout: List[Dict[str, float]]) -> np.ndarray:
        """
        Produce a compact feature vector for an optional learned refinement head.
        Layout-invariant summary statistics (mean/max/min/diff).
        """
        # Collect arrays
        sym = np.array([d["symmetry"] for d in per_layout], dtype=np.float32)
        brg = np.array([d["bridge_proxy"] for d in per_layout], dtype=np.float32)
        gap = np.array([d["gap_bucket"] for d in per_layout], dtype=np.float32)

        def stats(v: np.ndarray) -> List[float]:
            if v.size == 0:
                return [0, 0, 0, 0]
            return [
                float(v.mean()),
                float(v.max()),
                float(v.min()),
                float(v.max() - v.min()),
            ]

        feat = np.array(
            stats(sym) + stats(brg) + stats(gap),
            dtype=np.float32,
        )
        # Safe scaling
        m = float(np.linalg.norm(feat)) + self.cfg.feat_eps
        return feat / m
