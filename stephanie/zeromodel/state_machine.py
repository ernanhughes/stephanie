# stephanie/zeromodel/state_machine.py

"""
VPM State Machine
-----------------
Lightweight primitives to:
  • represent VPM goals/states
  • compute φ (feature) metrics from VPM tensors
  • execute VisualThought operations (zoom/bbox/path/highlight/blur)
  • evaluate utility deltas and benefit–cost score (BCS)

Shapes
------
VPM "image" X is a numpy array shaped [C, H, W] with uint8 or float32 values.
All internal ops standardize to float32 in [0, 1] during processing and return
to the original dtype/range on output.

External Dependencies
---------------------
- numpy
- PIL (for fast, portable resize & drawing)
- stephanie.utils.visual_thought (VisualThoughtOp, VisualThoughtType)

This module is intentionally dependency-light and pure-Python friendly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from stephanie.components.nexus.utils.visual_thought import VisualThoughtOp, VisualThoughtType


# ---------------------------------------------------------------------
# Goal & State
# ---------------------------------------------------------------------
@dataclass
class VPMGoal:
    """
    Goal specifies a weighted linear utility over φ metrics.
    Example:
        VPMGoal(weights={"separability": 1.0, "bridge_proxy": -1.0}, task_type="bottleneck_detection")
    """
    weights: Mapping[str, float]
    task_type: str = "generic"
    # Optional cost regularizer (per cumulative op-cost unit)
    cost_lambda: float = 0.0


@dataclass
class VPMState:
    """
    VPMState captures the current VPM tensor, derived metrics, and cumulative cost.

    Attributes
    ----------
    X : np.ndarray            [C, H, W] uint8/float32
    meta : Dict[str, Any]     auxiliary info (layout positions, cache flags, etc.)
    phi : Dict[str, float]    derived metrics computed from X (+ meta)
    goal : VPMGoal
    cost_accum : float        cumulative action cost applied so far
    dtype_range : Tuple[float, float]  original (min, max) to preserve scaling on write-back
    """
    X: np.ndarray
    meta: Dict[str, Any]
    phi: Dict[str, float]
    goal: VPMGoal
    cost_accum: float = 0.0
    dtype_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))

    @property
    def utility(self) -> float:
        """
        Utility is a linear combination of φ with optional cost penalty:
            U = Σ_k w_k * φ_k  - cost_lambda * cost_accum
        Missing φ_k are treated as 0.
        """
        u = 0.0
        for k, w in self.goal.weights.items():
            u += w * float(self.phi.get(k, 0.0))
        if self.goal.cost_lambda > 0:
            u -= self.goal.cost_lambda * float(self.cost_accum)
        return float(u)

    def clone(self) -> "VPMState":
        return VPMState(
            X=self.X.copy(),
            meta=dict(self.meta),
            phi=dict(self.phi),
            goal=self.goal,
            cost_accum=self.cost_accum,
            dtype_range=self.dtype_range,
        )


# ---------------------------------------------------------------------
# φ (Phi) Metrics
# ---------------------------------------------------------------------
def _as_float01(x: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Convert arbitrary dtype array into float32 in [0, 1] with range recorded.
    Accepts [C,H,W] or [H,W].
    """
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0, (0.0, 255.0)
    # Fallback: min-max to [0,1]
    xmin = float(np.min(x)) if x.size else 0.0
    xmax = float(np.max(x)) if x.size else 1.0
    rng = xmax - xmin if xmax > xmin else 1.0
    return ((x.astype(np.float32) - xmin) / rng), (xmin, xmax)


def _to_dtype_range(x01: np.ndarray, rng: Tuple[float, float], dtype) -> np.ndarray:
    """
    Map float32 [0,1] back to original numeric range & dtype.
    """
    lo, hi = rng
    if hi - lo <= 0:
        return (x01 * 0).astype(dtype)
    out = (x01 * (hi - lo) + lo)
    if dtype == np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    else:
        out = out.astype(dtype)
    return out


def _left_right_symmetry(ch: np.ndarray) -> float:
    """
    Pixelwise symmetry score between left and right halves of a single channel in [0,1].
    1.0 = perfectly symmetric; 0.0 = maximally asymmetric.
    """
    H, W = ch.shape
    L = ch[:, : W // 2]
    R = ch[:, W - (W // 2) :][:, ::-1]  # mirror right half
    N = min(L.shape[1], R.shape[1])
    if N == 0:
        return 0.0
    diff = np.abs(L[:, :N] - R[:, :N]).mean()
    return float(1.0 - diff)  # higher is more symmetric


def _bimodal_separability(ch: np.ndarray) -> float:
    """
    Rough separability proxy: project channel to X-axis, evaluate bimodality via
    "two-hump" measure. Returns [0,1], higher means more clearly separated modes.
    """
    H, W = ch.shape
    proj = ch.mean(axis=0)  # [W]
    if W < 8:
        return 0.0
    # Smooth a bit and find two peaks vs valley
    k = max(1, W // 64)
    if k > 1:
        kernel = np.ones(k, dtype=np.float32) / k
        proj = np.convolve(proj, kernel, mode="same")
    # Split halves and take max peaks
    mid = W // 2
    left_max = float(proj[:mid].max() if mid > 0 else 0.0)
    right_max = float(proj[mid:].max() if W - mid > 0 else 0.0)
    valley = float(proj[mid - k : mid + k].mean() if k > 0 else proj[mid])
    peak = 0.5 * (left_max + right_max)
    # Normalize contrast
    if peak <= 1e-6:
        return 0.0
    score = (peak - valley) / (peak + 1e-6)
    return float(np.clip(score, 0.0, 1.0))


def _bridge_proxy(ch: np.ndarray) -> float:
    """
    Bridge/bottleneck proxy: density at center vertical band relative to global.
    Higher means likely a thin connection region (i.e., risk ↑).
    """
    H, W = ch.shape
    if W < 8:
        return 0.0
    band_w = max(2, W // 16)
    mid_l = (W - band_w) // 2
    mid_r = mid_l + band_w
    center = float(ch[:, mid_l:mid_r].mean())
    global_mean = float(ch.mean())
    # Normalize by global magnitude; clamp to [0,1]
    if global_mean <= 1e-6:
        return 0.0
    return float(np.clip(center / (global_mean + 1e-6), 0.0, 1.0))


def _crossings_proxy(ed_ch: np.ndarray, threshold: float = 0.5) -> int:
    """
    Very rough "edge crossings" proxy using binarized edge density and counting
    connected runs across the center band. Larger counts ≈ more tangled edges.
    """
    H, W = ed_ch.shape
    if W < 8:
        return 0
    band_w = max(2, W // 16)
    mid_l = (W - band_w) // 2
    mid_r = mid_l + band_w
    stripe = (ed_ch[:, mid_l:mid_r] > threshold).astype(np.uint8)  # [H, band_w]
    # Count transitions per row and sum
    trans = np.abs(np.diff(stripe, axis=1)).sum(axis=1)  # [H]
    return int(trans.sum())


def compute_phi(vpm: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Compute a compact set of structural metrics from VPM channels + meta.
    Assumes typical channel semantics:
        0: node density
        1: edge density (optional)
        2: degree/centrality heat (optional)
    If channels are missing, proxies degrade gracefully.

    Returns
    -------
    Dict[str, float] with keys like:
        - separability      [0,1]   (higher = better separated communities)
        - vision_symmetry   [0,1]   (higher = more symmetric)
        - bridge_proxy      [0,1]   (higher = stronger thin connection)
        - spectral_gap      [0,1]   (if provided in meta; else coarse proxy)
        - crossings         int     (edge tangle proxy)
    """
    meta = meta or {}
    X = np.asarray(vpm)
    assert X.ndim in (2, 3), f"VPM must be [C,H,W] or [H,W]; got {X.shape}"

    if X.ndim == 2:
        X = X[None, ...]  # [1,H,W]

    X01, _rng = _as_float01(X)  # [C,H,W] in [0,1]
    C, H, W = X01.shape

    node = X01[0]
    edge = X01[1] if C > 1 else node
    heat = X01[2] if C > 2 else node

    # Core metrics
    separability = _bimodal_separability(node)
    vision_symmetry = _left_right_symmetry(node)
    bridge = _bridge_proxy(edge)
    crossings = _crossings_proxy(edge, threshold=0.5)

    # Spectral gap: trust meta if present, otherwise heuristic via "coherence"
    if "spectral_gap" in meta:
        spectral_gap = float(meta["spectral_gap"])
        # Map to [0,1] if it's raw (best-effort)
        if spectral_gap < 0 or spectral_gap > 1:
            spectral_gap = float(1.0 - math.exp(-abs(spectral_gap)))
    else:
        # Coarse proxy: more separable + symmetric → larger "gap"
        spectral_gap = float(np.clip(0.5 * separability + 0.5 * vision_symmetry, 0.0, 1.0))

    return {
        "separability": float(separability),
        "vision_symmetry": float(vision_symmetry),
        "bridge_proxy": float(bridge),
        "spectral_gap": float(spectral_gap),
        "crossings": int(crossings),
    }


# ---------------------------------------------------------------------
# Thought + Executor
# ---------------------------------------------------------------------
@dataclass
class Thought:
    """
    A single reasoning step comprised of one or more visual operations.
    'cost' is a nominal compute/latency budget unit (tuned in your pipeline).
    """
    name: str
    ops: List[VisualThoughtOp]
    intent: Optional[str] = None
    cost: float = 0.0


class ThoughtExecutor:
    """
    Applies visual thoughts to VPM states and scores their impact.

    visual_op_cost: per-op additive cost map (e.g., {"zoom":1.0, "bbox":0.3, ...})
    """

    def __init__(self, visual_op_cost: Optional[Mapping[str, float]] = None):
        self.visual_op_cost = dict(visual_op_cost or {})

    # ---- Public API -------------------------------------------------
    def score_thought(
        self,
        state: VPMState,
        thought: Thought,
        *,
        recompute_phi: bool = True,
    ) -> Tuple[VPMState, float, float, float]:
        """
        Apply 'thought' to 'state' and return:
            new_state, delta_utility, total_cost, bcs  (benefit - cost)

        bcs is simply (new.utility - old.utility) - total_cost
        """
        old_u = state.utility
        out = state.clone()

        # Apply ops
        total_cost = float(thought.cost)
        for op in thought.ops:
            out.X = self._apply_op(out.X, op)
            total_cost += float(self.visual_op_cost.get(op.type.value, 0.0))

        # Recompute φ and utility
        if recompute_phi:
            out.phi = compute_phi(out.X, out.meta)
        out.cost_accum = state.cost_accum + total_cost
        new_u = out.utility

        delta = float(new_u - old_u)
        bcs = float(delta - total_cost)
        return out, delta, total_cost, bcs

    # ---- Visual Ops -------------------------------------------------
    def _apply_op(self, X: np.ndarray, op: VisualThoughtOp) -> np.ndarray:
        """
        Apply a single VisualThoughtOp to a [C,H,W] array. Returns new array.
        """
        X = np.asarray(X)
        assert X.ndim in (2, 3), f"VPM must be [C,H,W] or [H,W]; got {X.shape}"
        was_2d = (X.ndim == 2)
        if was_2d:
            X = X[None, ...]  # [1,H,W]

        dtype = X.dtype
        X01, rng = _as_float01(X)

        if op.type == VisualThoughtType.ZOOM:
            X01 = self._op_zoom(X01, op.params)
        elif op.type == VisualThoughtType.BBOX:
            X01 = self._op_bbox(X01, op.params)
        elif op.type == VisualThoughtType.PATH:
            X01 = self._op_path(X01, op.params)
        elif op.type == VisualThoughtType.HIGHLIGHT:
            X01 = self._op_highlight(X01, op.params)
        elif op.type == VisualThoughtType.BLUR:
            X01 = self._op_blur(X01, op.params)
        else:
            # Unknown op: no-op
            pass

        X_out = _to_dtype_range(X01, rng, dtype)
        if was_2d:
            X_out = X_out[0]
        return X_out

    # ---- Concrete op implementations -------------------------------
    def _op_zoom(self, X01: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Zoom into a region around 'center' with a given 'scale' (>1 zoom-in).
        params:
            center: (cx, cy) in pixel coords of the HxW plane
            scale:  float, >= 1.0
        """
        C, H, W = X01.shape
        cx = int(params.get("center", (W // 2, H // 2))[0])
        cy = int(params.get("center", (W // 2, H // 2))[1])
        scale = float(params.get("scale", 2.0))
        scale = max(1.0, float(scale))

        # Compute crop box
        crop_w = max(2, int(round(W / scale)))
        crop_h = max(2, int(round(H / scale)))
        x1 = int(np.clip(cx - crop_w // 2, 0, W - crop_w))
        y1 = int(np.clip(cy - crop_h // 2, 0, H - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Crop & resize each channel with PIL for simplicity/quality
        out = np.zeros_like(X01)
        for c in range(C):
            ch = (X01[c] * 255).astype(np.uint8)
            pil = Image.fromarray(ch, mode="L")
            crop = pil.crop((x1, y1, x2, y2))
            resized = crop.resize((W, H), resample=Image.BICUBIC)
            out[c] = np.asarray(resized, dtype=np.float32) / 255.0
        return out

    def _op_bbox(self, X01: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Draw an outline box to emphasize a region. Rendering is additive and clipped.
        params:
            xyxy: (x1,y1,x2,y2)
            width: int (line width)
            intensity: float in [0,1] (how bright the box strokes)
            channel: optional int to draw only on that channel; default overlays all
        """
        C, H, W = X01.shape
        x1, y1, x2, y2 = map(int, params.get("xyxy", (W//4, H//4, 3*W//4, 3*H//4)))
        width = int(params.get("width", 2))
        intensity = float(np.clip(params.get("intensity", 1.0), 0.0, 1.0))
        channel = params.get("channel", None)

        if channel is not None:
            channels = [int(channel)]
        else:
            channels = list(range(C))

        for c in channels:
            base = (X01[c] * 255).astype(np.uint8)
            pil = Image.fromarray(base, mode="L")
            draw = ImageDraw.Draw(pil)
            for w in range(width):
                draw.rectangle(
                    (x1 - w, y1 - w, x2 + w, y2 + w),
                    outline=int(max(0, min(255, int(255 * intensity)))),
                )
            X01[c] = np.asarray(pil, dtype=np.float32) / 255.0
        return X01

    def _op_path(self, X01: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Draw a polyline (optionally with arrows) to indicate flow/route.
        params:
            points: List[(x,y)]
            width: int
            intensity: float in [0,1]
            channel: optional int
        """
        C, H, W = X01.shape
        pts = params.get("points", None)
        if not pts or len(pts) < 2:
            return X01
        width = int(params.get("width", 2))
        intensity = float(np.clip(params.get("intensity", 1.0), 0.0, 1.0))
        channel = params.get("channel", None)

        if channel is not None:
            channels = [int(channel)]
        else:
            channels = list(range(C))

        for c in channels:
            base = (X01[c] * 255).astype(np.uint8)
            pil = Image.fromarray(base, mode="L")
            draw = ImageDraw.Draw(pil)
            draw.line(pts, fill=int(255 * intensity), width=width)
            X01[c] = np.asarray(pil, dtype=np.float32) / 255.0
        return X01

    def _op_highlight(self, X01: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Fill a polygon or rectangular region with semi-opaque emphasis.
        params:
            polygon: List[(x,y)]  OR  xyxy: (x1,y1,x2,y2)
            opacity: float in [0,1]
            channel: optional int
        """
        C, H, W = X01.shape
        opacity = float(np.clip(params.get("opacity", 0.35), 0.0, 1.0))
        channel = params.get("channel", None)
        color_val = int(255 * opacity)

        if "polygon" in params and params["polygon"]:
            poly = params["polygon"]
            def draw_on(pil):
                overlay = Image.new("L", pil.size, 0)
                d = ImageDraw.Draw(overlay)
                d.polygon(poly, fill=color_val)
                return Image.composite(Image.new("L", pil.size, 255), pil, overlay)
        else:
            x1, y1, x2, y2 = map(int, params.get("xyxy", (W//3, H//3, 2*W//3, 2*H//3)))
            def draw_on(pil):
                overlay = Image.new("L", pil.size, 0)
                d = ImageDraw.Draw(overlay)
                d.rectangle((x1, y1, x2, y2), fill=color_val)
                return Image.composite(Image.new("L", pil.size, 255), pil, overlay)

        channels = [int(channel)] if channel is not None else list(range(C))
        for c in channels:
            base = (X01[c] * 255).astype(np.uint8)
            pil = Image.fromarray(base, mode="L")
            out = draw_on(pil)
            X01[c] = np.asarray(out, dtype=np.float32) / 255.0
        return X01

    def _op_blur(self, X01: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply Gaussian blur to a region (or whole image if no region).
        params:
            xyxy: optional (x1,y1,x2,y2) region to blur
            radius: float (blur radius)
            channel: optional int
        """
        C, H, W = X01.shape
        radius = float(max(0.1, params.get("radius", 1.5)))
        xyxy = params.get("xyxy", None)
        channel = params.get("channel", None)
        channels = [int(channel)] if channel is not None else list(range(C))

        for c in channels:
            base = (X01[c] * 255).astype(np.uint8)
            pil = Image.fromarray(base, mode="L")
            if xyxy is None:
                blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
                X01[c] = np.asarray(blurred, dtype=np.float32) / 255.0
            else:
                x1, y1, x2, y2 = map(int, xyxy)
                # Extract region, blur, paste back
                crop = pil.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=radius))
                pil.paste(crop, (x1, y1))
                X01[c] = np.asarray(pil, dtype=np.float32) / 255.0
        return X01
