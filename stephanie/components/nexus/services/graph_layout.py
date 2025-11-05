# stephanie/components/nexus/services/graph_layout.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import networkx as nx
import numpy as np

LayoutName = Literal["forceatlas2", "spectral"]
ChannelName = Literal["node_density", "edge_density", "degree_heatmap"]

# --------------------------- Config ---------------------------

@dataclass
class LayoutConfig:
    img_size: int = 256
    channels: Tuple[ChannelName, ...] = ("node_density", "edge_density", "degree_heatmap")
    cache_dir: Optional[str] = None
    # Layout knobs
    fa2_k: Optional[float] = None           # nx.spring_layout k
    fa2_iter: int = 200
    spectral_center: Optional[Tuple[float, float]] = None
    seed: int = 42
    # Rasterization knobs
    node_sigma: float = 1.5                 # pixel sigma for node splats
    edge_thickness: float = 0.75            # Bresenham/anti-aliased thickness in px
    degree_gamma: float = 0.75              # gamma for degree heat
    # Performance knobs
    max_edges_for_crossings: int = 6000     # skip crossing estimate above this

# --------------------------- Public API ---------------------------

def render_multi_layout_vpm(
    G: nx.Graph,
    layouts: Iterable[LayoutName] = ("forceatlas2", "spectral"),
    config: Optional[Dict] = None,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Return:
      vpms: List[np.uint8 array] each with shape [C, H, W] (channels last? -> channels first)
            *channels first* => [C, H, W]
      metas: List[dict] with keys:
        - positions: {node_id (str): [x,y] in [0,1]}
        - layout: layout name
        - layout_hash: hex
        - layout_fallback: Optional[str]
        - spectral_gap: float
        - crossings: int (approx)
        - img_size: int
        - channel_names: List[str]
    """
    cfg = _load_cfg(config)
    H = W = int(cfg.img_size)

    # Build a graph signature for caching
    g_sig = _graph_signature(G)

    vpms: List[np.ndarray] = []
    metas: List[Dict] = []

    for layout in layouts:
        cache_key = _cache_key(
            g_sig=g_sig,
            layout=layout,
            size=cfg.img_size,
            channels=cfg.channels,
            knobs={
                "fa2_k": cfg.fa2_k,
                "fa2_iter": cfg.fa2_iter,
                "deg_gamma": cfg.degree_gamma,
                "node_sigma": cfg.node_sigma,
                "edge_thickness": cfg.edge_thickness,
            },
        )
        if cfg.cache_dir:
            cached = _try_cache_load(cfg.cache_dir, cache_key)
            if cached is not None:
                vpm, meta = cached
                vpms.append(vpm)
                metas.append(meta)
                continue

        # Compute layout positions
        pos, fallback, layout_hash = _compute_positions(G, layout, cfg)

        # Normalize -> [0,1] square
        pos01 = _normalize_positions(pos)

        # Channels
        channels = []
        if "node_density" in cfg.channels:
            ch = _rasterize_nodes(G, pos01, H, W, sigma=cfg.node_sigma)
            channels.append(ch)
        if "edge_density" in cfg.channels:
            ch = _rasterize_edges(G, pos01, H, W, thickness=cfg.edge_thickness)
            channels.append(ch)
        if "degree_heatmap" in cfg.channels:
            ch = _degree_heatmap(G, H, W, gamma=cfg.degree_gamma)
            channels.append(_apply_positions_mask(ch, pos01, H, W))

        # Stack and scale to uint8 [C, H, W]
        if len(channels) == 0:
            # Always deliver at least one channel
            channels = [np.zeros((H, W), dtype=np.float32)]
        vpm_f32 = np.stack(channels, axis=0)  # [C, H, W]
        vpm_u8 = _to_uint8(vpm_f32)

        # Meta
        spectral_gap = _safe_spectral_gap(G)
        crossings = _approx_crossings(pos01, G, limit=cfg.max_edges_for_crossings)

        meta = {
            "positions": {str(k): [float(v[0]), float(v[1])] for k, v in pos01.items()},
            "layout": layout,
            "layout_hash": layout_hash,
            "layout_fallback": fallback,
            "spectral_gap": float(spectral_gap),
            "crossings": int(crossings),
            "img_size": cfg.img_size,
            "channel_names": list(cfg.channels),
        }

        # Cache
        if cfg.cache_dir:
            _cache_save(cfg.cache_dir, cache_key, vpm_u8, meta)

        vpms.append(vpm_u8)
        metas.append(meta)

    return vpms, metas

# --------------------------- Layout computation ---------------------------

def _compute_positions(
    G: nx.Graph,
    layout: LayoutName,
    cfg: LayoutConfig,
) -> Tuple[Dict, Optional[str], str]:
    rng = np.random.RandomState(cfg.seed)
    fallback: Optional[str] = None

    if layout == "forceatlas2":
        # Use spring_layout tuned to act like FA2
        try:
            k = cfg.fa2_k
            if k is None:
                # Heuristic: optimal k ~ 1/sqrt(n)
                n = max(1, G.number_of_nodes())
                k = 1.0 / np.sqrt(n)
            pos = nx.spring_layout(
                G,
                k=k,
                iterations=cfg.fa2_iter,
                seed=cfg.seed,
                weight="weight",
                dim=2,
            )
        except Exception:
            # Fallback to FR or Kamada-Kawai if spring fails
            try:
                pos = nx.fruchterman_reingold_layout(G, seed=cfg.seed, dim=2)
                fallback = "fruchterman_reingold"
            except Exception:
                pos = nx.kamada_kawai_layout(G, dim=2)
                fallback = "kamada_kawai"

    elif layout == "spectral":
        try:
            pos = nx.spectral_layout(G, dim=2, center=cfg.spectral_center)
        except Exception:
            # Fallback to spring
            pos = nx.spring_layout(G, seed=cfg.seed, dim=2)
            fallback = "spring"

    else:
        # Unknown layout -> spring with fallback flag
        pos = nx.spring_layout(G, seed=cfg.seed, dim=2)
        fallback = "spring"

    # Hash the positions for reproducibility/auditing
    layout_hash = _positions_hash(pos)
    return pos, fallback, layout_hash

def _normalize_positions(pos: Dict) -> Dict:
    # Convert to array
    P = np.array(list(pos.values()), dtype=np.float64)
    if P.size == 0:
        return {}
    minv = P.min(axis=0)
    maxv = P.max(axis=0)
    span = np.maximum(maxv - minv, 1e-9)
    P01 = (P - minv) / span
    keys = list(pos.keys())
    return {keys[i]: (float(P01[i, 0]), float(P01[i, 1])) for i in range(len(keys))}

def _positions_hash(pos: Dict) -> str:
    items = sorted((str(k), float(v[0]), float(v[1])) for k, v in pos.items())
    raw = json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

# --------------------------- Channels ---------------------------

def _rasterize_nodes(
    G: nx.Graph,
    pos01: Dict,
    H: int,
    W: int,
    sigma: float = 1.5,
) -> np.ndarray:
    """Gaussian node splats (fast separable)."""
    out = np.zeros((H, W), dtype=np.float32)
    if not pos01:
        return out
    # Precompute Gaussian kernel (odd size)
    rad = max(1, int(3 * sigma))
    xs = np.arange(-rad, rad + 1)
    kernel_1d = np.exp(-(xs**2) / (2 * sigma * sigma)).astype(np.float32)
    kernel_1d /= (kernel_1d.sum() + 1e-9)
    k = kernel_1d
    for n, (x, y) in pos01.items():
        i = int(np.clip(y * (H - 1), 0, H - 1))
        j = int(np.clip(x * (W - 1), 0, W - 1))
        # Separable add
        _add_gaussian(out, i, j, k)
    out = out / (out.max() + 1e-9)
    return out

def _add_gaussian(img: np.ndarray, i: int, j: int, k: np.ndarray):
    H, W = img.shape
    rad = len(k) // 2
    # Vertical
    top = max(0, i - rad)
    bot = min(H, i + rad + 1)
    kv_top = rad - (i - top)
    kv_bot = kv_top + (bot - top)
    img[top:bot, j] += k[kv_top:kv_bot]
    # Horizontal (convolve the vertical line)
    left = max(0, j - rad)
    right = min(W, j + rad + 1)
    kh_left = rad - (j - left)
    kh_right = kh_left + (right - left)
    # Broadcasted add across row window
    img[i, left:right] += k[kh_left:kh_right]

def _rasterize_edges(
    G: nx.Graph,
    pos01: Dict,
    H: int,
    W: int,
    thickness: float = 0.75,
) -> np.ndarray:
    """Simple Xiaolin Wu style anti-aliased lines (approx)."""
    out = np.zeros((H, W), dtype=np.float32)
    if not pos01:
        return out
    t = max(0.5, float(thickness))

    for u, v, data in G.edges(data=True):
        if u not in pos01 or v not in pos01:
            continue
        x0, y0 = pos01[u]
        x1, y1 = pos01[v]
        r0 = int(np.clip(y0 * (H - 1), 0, H - 1))
        c0 = int(np.clip(x0 * (W - 1), 0, W - 1))
        r1 = int(np.clip(y1 * (H - 1), 0, H - 1))
        c1 = int(np.clip(x1 * (W - 1), 0, W - 1))
        _draw_aa_line(out, r0, c0, r1, c1, t)
    out = out / (out.max() + 1e-9)
    return out

def _draw_aa_line(img: np.ndarray, r0: int, c0: int, r1: int, c1: int, t: float):
    """Rasterize anti-aliased line with thickness t (in pixels)."""
    H, W = img.shape
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        _splat_disk(img, r, c, t)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

def _splat_disk(img: np.ndarray, r: int, c: int, t: float):
    H, W = img.shape
    rad = int(max(1, np.ceil(t)))
    rr = np.arange(r - rad, r + rad + 1)
    cc = np.arange(c - rad, c + rad + 1)
    rr = rr[(0 <= rr) & (rr < H)]
    cc = cc[(0 <= cc) & (cc < W)]
    if rr.size == 0 or cc.size == 0:
        return
    R, C = np.meshgrid(rr, cc, indexing="ij")
    dist = np.sqrt((R - r) ** 2 + (C - c) ** 2)
    mask = (dist <= t).astype(np.float32)
    img[np.ix_(rr, cc)] += mask * (1.0 / (t + 1e-6))

def _degree_heatmap(G: nx.Graph, H: int, W: int, gamma: float = 0.75) -> np.ndarray:
    """Global scalar mapped everywhere (then masked by positions)."""
    degs = np.array([d for _, d in G.degree()], dtype=np.float32)
    if degs.size == 0:
        val = 0.0
    else:
        v = (degs - degs.min()) / (max(1e-6, degs.max() - degs.min()))
        val = float(np.power(v.mean(), gamma))
    ch = np.full((H, W), val, dtype=np.float32)
    return ch

def _apply_positions_mask(ch: np.ndarray, pos01: Dict, H: int, W: int) -> np.ndarray:
    # Downweight empty areas slightly to avoid flat planes
    if not pos01:
        return ch
    mask = np.zeros_like(ch)
    for _, (x, y) in pos01.items():
        i = int(np.clip(y * (H - 1), 0, H - 1))
        j = int(np.clip(x * (W - 1), 0, W - 1))
        _splat_disk(mask, i, j, 2.0)
    mask = (mask > 0).astype(np.float32)
    return 0.7 * ch + 0.3 * (ch * mask)

def _to_uint8(stack_f32: np.ndarray) -> np.ndarray:
    # Normalize each channel independently
    out = np.empty_like(stack_f32, dtype=np.uint8)
    for ci in range(stack_f32.shape[0]):
        ch = stack_f32[ci]
        m, M = float(ch.min()), float(ch.max())
        if M - m < 1e-9:
            out[ci] = np.zeros_like(ch, dtype=np.uint8)
        else:
            out[ci] = np.clip(255.0 * (ch - m) / (M - m), 0, 255).astype(np.uint8)
    return out

# --------------------------- Metrics ---------------------------

def _safe_spectral_gap(G: nx.Graph) -> float:
    """Algebraic connectivity (λ2) on Laplacian; returns 0.0 if ill-defined."""
    try:
        if G.number_of_nodes() <= 1:
            return 0.0
        # Use SciPy if available via nx; else fallback to dense eig (small graphs)
        L = nx.laplacian_matrix(G).astype(float)
        try:
            # Prefer ARPACK if present
            import scipy.sparse.linalg as sla  # type: ignore
            vals = sla.eigsh(L, k=2, which="SM", return_eigenvectors=False)
            vals = np.sort(np.real(vals))
        except Exception:
            # Dense fallback (small-ish graphs)
            Ld = L.toarray()
            vals = np.linalg.eigvalsh(Ld)
            vals = np.sort(vals)
        if vals.size < 2:
            return 0.0
        return float(max(0.0, vals[1]))
    except Exception:
        return 0.0

def _approx_crossings(
    pos01: Dict,
    G: nx.Graph,
    limit: int = 6000,
) -> int:
    """O(E^2) crossing counter with early cap for larger graphs."""
    E = list(G.edges())
    m = len(E)
    if m == 0 or not pos01:
        return 0
    if m > limit:
        return -1  # skip
    # Segment intersection test
    def seg(p, q, r, s) -> bool:
        # p=(x1,y1) q=(x2,y2), r=(x3,y3) s=(x4,y4)
        # Exclude shared endpoints
        if p == r or p == s or q == r or q == s:
            return False
        return _segments_intersect(p, q, r, s)

    crossings = 0
    for i in range(m):
        u, v = E[i]
        if u not in pos01 or v not in pos01:
            continue
        p = pos01[u]
        q = pos01[v]
        for j in range(i + 1, m):
            a, b = E[j]
            if a not in pos01 or b not in pos01:
                continue
            r = pos01[a]
            s = pos01[b]
            if seg(p, q, r, s):
                crossings += 1
    return crossings

def _segments_intersect(p, q, r, s) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    def on_seg(a, b, c):
        return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and
                min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)

    o1 = orient(p, q, r)
    o2 = orient(p, q, s)
    o3 = orient(r, s, p)
    o4 = orient(r, s, q)

    if o1 == 0 and on_seg(p, q, r): return True
    if o2 == 0 and on_seg(p, q, s): return True
    if o3 == 0 and on_seg(r, s, p): return True
    if o4 == 0 and on_seg(r, s, q): return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

# --------------------------- Caching ---------------------------

def _graph_signature(G: nx.Graph) -> str:
    """Hash nodes + edges (+ weights if present)."""
    nodes = sorted([str(n) for n in G.nodes()])
    edges = sorted([(*sorted((str(u), str(v))), float(G[u][v].get("weight", 1.0))) for u, v in G.edges()])
    raw = json.dumps({"n": nodes, "e": edges}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def _cache_key(
    g_sig: str,
    layout: str,
    size: int,
    channels: Tuple[str, ...],
    knobs: Dict,
) -> str:
    payload = {
        "g": g_sig,
        "l": layout,
        "s": size,
        "c": list(channels),
        "k": knobs,
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def _try_cache_load(cache_dir: str, key: str) -> Optional[Tuple[np.ndarray, Dict]]:
    root = Path(cache_dir)
    vpm_p = root / f"{key}.npy"
    meta_p = root / f"{key}.json"
    try:
        if vpm_p.exists() and meta_p.exists():
            arr = np.load(vpm_p, allow_pickle=False)
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Sanity: uint8, [C,H,W]
            if arr.dtype == np.uint8 and arr.ndim == 3:
                return arr, meta
    except Exception:
        return None
    return None

def _cache_save(cache_dir: str, key: str, vpm: np.ndarray, meta: Dict) -> None:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"{key}.npy", vpm, allow_pickle=False)
    with open(root / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))

# --------------------------- Config loader ---------------------------

def _load_cfg(config: Optional[Dict]) -> LayoutConfig:
    if config is None:
        return LayoutConfig()
    lc = LayoutConfig()
    for k, v in (config or {}).items():
        if hasattr(lc, k):
            setattr(lc, k, v)
    return lc
