# stephanie/utils/visual_thought.py
"""
Visual Thought primitives and executor.

This module defines:
  - VisualThoughtType: the catalog of visual ops
  - VisualThoughtOp: a concrete, serializable op
  - Thought: a named bundle of 1..N ops with intent + cost
  - ThoughtExecutor: deterministic raster engine that applies ops to a VPM

VPM shape convention: [C, H, W]
  * Values may be uint8 (0..255) or float32 (0..1). We preserve dtype on output.
  * Ops are applied on copies (never in-place on the input array).

Returned scoring tuple from executor:
  (new_state: VPMState, delta: float, cost: float, bcs: float)

Where:
  - delta = new_state.utility - old_state.utility
  - cost  = sum(op costs) or provided Thought.cost
  - bcs   = benefit-cost score = max(delta - λ * cost, 0), λ∈[0,1] (configurable)

This module is dependency-light. For polygon masks and crisp outlines it uses PIL.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# Optional but recommended for polygon masking and simple text/line drawing
try:
    from PIL import Image, ImageDraw  # type: ignore
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False

# These imports are part of your project
# VPMState contains (X: np.ndarray [C,H,W], meta: dict, phi: dict, goal: VPMGoal, utility: float)
# compute_phi recomputes metrics from VPM + meta (e.g., separability, symmetry, bridge_proxy ...)
from stephanie.zeromodel.state_machine import VPMState, VPMGoal, compute_phi


# --------------------------------------------------------------------------------------
# Visual Ops
# --------------------------------------------------------------------------------------

class VisualThoughtType(str, enum.Enum):
    ZOOM = "zoom"           # params: center(x:int,y:int), scale(float>=1)
    BBOX = "bbox"           # params: xyxy(x1:int,y1:int,x2:int,y2:int), thickness:int=2, boost:float
    PATH = "path"           # params: points[List[Tuple[int,int]]], arrows:bool=False, thickness:int=2
    HIGHLIGHT = "highlight" # params: polygon[List[Tuple[int,int]]], opacity:float in [0,1], boost:float
    BLUR = "blur"           # params: xyxy, k:int (odd kernel size), passes:int=1


@dataclass(frozen=True)
class VisualThoughtOp:
    type: VisualThoughtType
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "params": self.params}

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "VisualThoughtOp":
        return VisualThoughtOp(VisualThoughtType(obj["type"]), dict(obj.get("params", {})))


@dataclass
class Thought:
    name: str
    ops: List[VisualThoughtOp]
    intent: Optional[str] = None
    cost: float = 0.0  # Optional override (otherwise derived from op-wise costs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "intent": self.intent,
            "cost": self.cost,
            "ops": [op.to_dict() for op in self.ops],
        }

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "Thought":
        return Thought(
            name=obj.get("name", "Thought"),
            intent=obj.get("intent"),
            cost=float(obj.get("cost", 0.0)),
            ops=[VisualThoughtOp.from_dict(x) for x in obj.get("ops", [])],
        )


# --------------------------------------------------------------------------------------
# Raster helpers (pure NumPy + optional PIL)
# --------------------------------------------------------------------------------------

def _ensure_float01(X: np.ndarray) -> Tuple[np.ndarray, Optional[np.dtype]]:
    """
    Convert to float32 in [0,1]. Returns (float_view, original_dtype or None).
    """
    orig_dtype = X.dtype
    Xf = X.astype(np.float32, copy=False)
    if np.issubdtype(orig_dtype, np.integer):
        Xf = Xf / 255.0
    Xf = np.clip(Xf, 0.0, 1.0)
    return Xf, orig_dtype


def _restore_dtype(Xf: np.ndarray, orig_dtype: Optional[np.dtype]) -> np.ndarray:
    if orig_dtype is None:
        return Xf
    if np.issubdtype(orig_dtype, np.integer):
        Y = np.clip(np.rint(Xf * 255.0), 0, 255).astype(orig_dtype)
        return Y
    return Xf.astype(orig_dtype)


def _clip_xyxy(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1c, y1c = max(0, min(W - 1, x1)), max(0, min(H - 1, y1))
    x2c, y2c = max(0, min(W - 1, x2)), max(0, min(H - 1, y2))
    if x2c < x1c:
        x1c, x2c = x2c, x1c
    if y2c < y1c:
        y1c, y2c = y2c, y1c
    return x1c, y1c, x2c, y2c


def _box_blur_region(ch: np.ndarray, x1: int, y1: int, x2: int, y2: int, k: int = 3, passes: int = 1) -> None:
    """
    In-place box blur of region [y1:y2+1, x1:x2+1]. k must be odd.
    """
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    r = k // 2
    for _ in range(max(1, int(passes))):
        roi = ch[y1 : y2 + 1, x1 : x2 + 1]
        # Pad reflect to handle borders
        padded = np.pad(roi, ((r, r), (r, r)), mode="reflect")
        # Integral image trick
        integral = padded.cumsum(axis=0).cumsum(axis=1)
        H, W = roi.shape
        # Compute summed area over kxk for each pixel
        sum_area = (
            integral[k:, k:]
            - integral[:-k, k:]
            - integral[k:, :-k]
            + integral[:-k, :-k]
        )
        roi[:] = sum_area / (k * k)


def _draw_rect(ch: np.ndarray, x1: int, y1: int, x2: int, y2: int, thickness: int = 2, value: float = 1.0) -> None:
    """In-place rectangle outline on channel ch in [0,1]."""
    H, W = ch.shape
    x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, W, H)
    t = max(1, int(thickness))
    ch[y1 : min(H, y1 + t), x1 : x2 + 1] = value
    ch[max(0, y2 - t + 1) : y2 + 1, x1 : x2 + 1] = value
    ch[y1 : y2 + 1, x1 : min(W, x1 + t)] = value
    ch[y1 : y2 + 1, max(0, x2 - t + 1) : x2 + 1] = value


def _draw_polyline(ch: np.ndarray, pts: Sequence[Tuple[int, int]], thickness: int = 2, value: float = 1.0) -> None:
    """In-place polyline (Bresenham) on channel ch in [0,1]."""
    t = max(1, int(thickness))
    H, W = ch.shape

    def draw_point(x, y):
        xs = slice(max(0, x - t // 2), min(W, x + t // 2 + 1))
        ys = slice(max(0, y - t // 2), min(H, y + t // 2 + 1))
        ch[ys, xs] = value

    def draw_line(x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            draw_point(x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        draw_line(int(x0), int(y0), int(x1), int(y1))


def _apply_zoom(vpm: np.ndarray, center: Tuple[int, int], scale: float) -> np.ndarray:
    """
    Zoom by cropping a window around `center` and resizing back via nearest-neighbor.
    scale >= 1.0 (1.0 = no-op). Larger = tighter crop = stronger zoom.
    """
    C, H, W = vpm.shape
    scale = float(max(1.0, scale))
    # Effective crop size
    crop_w = int(round(W / scale))
    crop_h = int(round(H / scale))
    cx = int(center[0])
    cy = int(center[1])
    x1 = max(0, min(W - crop_w, cx - crop_w // 2))
    y1 = max(0, min(H - crop_h, cy - crop_h // 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    crop = vpm[:, y1:y2, x1:x2]  # [C, h, w]

    # Nearest-neighbor upsample back to [H,W] without external deps
    # Compute integer stride
    if crop_w == 0 or crop_h == 0:
        return vpm.copy()
    sy = max(1, H // crop.shape[1])
    sx = max(1, W // crop.shape[2])
    up = np.repeat(np.repeat(crop, sy, axis=1), sx, axis=2)
    # If slightly off, center-crop/pad to exact HxW
    upH, upW = up.shape[1], up.shape[2]
    # center-crop or pad
    if upH >= H:
        ystart = (upH - H) // 2
        up = up[:, ystart : ystart + H, :]
    else:
        pad_top = (H - upH) // 2
        pad_bot = H - upH - pad_top
        up = np.pad(up, ((0, 0), (pad_top, pad_bot), (0, 0)), mode="edge")
    if upW >= W:
        xstart = (upW - W) // 2
        up = up[:, :, xstart : xstart + W]
    else:
        pad_l = (W - upW) // 2
        pad_r = W - upW - pad_l
        up = np.pad(up, ((0, 0), (0, 0), (pad_l, pad_r)), mode="edge")
    return up


def _apply_bbox(vpm: np.ndarray, xyxy: Tuple[int, int, int, int], thickness: int = 2, boost: float = 0.15) -> None:
    """
    Draw rectangle and slightly boost channel-0 contrast inside it.
    """
    C, H, W = vpm.shape
    x1, y1, x2, y2 = _clip_xyxy(*xyxy, W=W, H=H)
    # Outline on channel 1 (edge density / overlay)
    ch_outline = min(1, 1)  # use channel-1 if exists
    ch = vpm[ch_outline if C > 1 else 0]
    _draw_rect(ch, x1, y1, x2, y2, thickness=thickness, value=1.0)

    # Boost channel-0 inside the box
    base = vpm[0]
    roi = base[y1 : y2 + 1, x1 : x2 + 1]
    roi[:] = np.clip(roi * (1.0 + float(boost)), 0.0, 1.0)


def _apply_path(vpm: np.ndarray, points: Sequence[Tuple[int, int]], arrows: bool = False, thickness: int = 2) -> None:
    """
    Draw a polyline on channel-2 (degree heat / overlay). If absent, fallback to channel-0.
    """
    C, H, W = vpm.shape
    ch_idx = 2 if C > 2 else 0
    ch = vpm[ch_idx]
    if not points or len(points) < 2:
        return
    _draw_polyline(ch, points, thickness=max(1, int(thickness)), value=1.0)
    # (Optional) arrowheads can be added with small triangles; omitted for simplicity.


def _apply_highlight(vpm: np.ndarray, polygon: Sequence[Tuple[int, int]], opacity: float = 0.25, boost: float = 0.25) -> None:
    """
    Fill polygon region with a translucent mask, boosting channel-0 under the mask.
    Requires PIL for robust polygon rasterization; otherwise uses convex hull AABB fallback.
    """
    C, H, W = vpm.shape
    opacity = float(np.clip(opacity, 0.0, 1.0))
    boost = float(max(0.0, boost))

    if _HAS_PIL and polygon:
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(list(map(tuple, polygon)), fill=int(opacity * 255))
        m = np.asarray(mask, dtype=np.float32) / 255.0
    else:  # fallback: highlight AABB of polygon
        xs = [x for x, _ in polygon] if polygon else [0, W - 1]
        ys = [y for _, y in polygon] if polygon else [0, H - 1]
        x1, y1, x2, y2 = _clip_xyxy(min(xs), min(ys), max(xs), max(ys), W=W, H=H)
        m = np.zeros((H, W), dtype=np.float32)
        m[y1 : y2 + 1, x1 : x2 + 1] = opacity

    # Apply on channel-0
    vpm[0] = np.clip(vpm[0] * (1.0 + boost * m), 0.0, 1.0)


def _apply_blur(vpm: np.ndarray, xyxy: Tuple[int, int, int, int], k: int = 3, passes: int = 1) -> None:
    """
    Mild de-emphasis by blurring all channels within the box.
    """
    C, H, W = vpm.shape
    x1, y1, x2, y2 = _clip_xyxy(*xyxy, W=W, H=H)
    for c in range(C):
        _box_blur_region(vpm[c], x1, y1, x2, y2, k=k, passes=passes)


# --------------------------------------------------------------------------------------
# Thought Executor
# --------------------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    # Simple op cost schedule (arbitrary units scaled by lambda_cost)
    visual_op_cost: Dict[str, float] = None
    lambda_cost: float = 0.2  # weight for cost in BCS = max(delta - λ * cost, 0)

    def __post_init__(self):
        if self.visual_op_cost is None:
            self.visual_op_cost = {
                "zoom": 1.0,
                "bbox": 0.3,
                "path": 0.4,
                "highlight": 0.5,
                "blur": 0.6,
            }


class ThoughtExecutor:
    """
    Deterministic raster engine that:
      - normalizes VPM to float[0,1]
      - applies ops in sequence
      - restores dtype
      - recomputes phi & utility via compute_phi + VPMState
      - scores benefit/cost

    Usage:
      new_state, delta, cost, bcs = executor.score_thought(state, thought)
    """

    def __init__(self, visual_op_cost: Optional[Dict[str, float]] = None, lambda_cost: float = 0.2):
        self.cfg = ExecutorConfig(visual_op_cost=visual_op_cost, lambda_cost=lambda_cost)

    # ----------------------------- public API -----------------------------

    def score_thought(self, state: VPMState, thought: Thought) -> Tuple[VPMState, float, float, float]:
        """
        Apply a Thought to a state, recompute metrics, and return scores.
        """
        vpm = state.X  # [C,H,W]
        vpm_f, orig_dtype = _ensure_float01(vpm.copy())

        # Apply ops in sequence
        for op in thought.ops:
            vpm_f = self._apply_op(vpm_f, op)

        # Restore dtype
        vpm_out = _restore_dtype(vpm_f, orig_dtype)

        # Recompute metrics/utility
        phi_new = compute_phi(vpm_out, state.meta)
        new_state = VPMState(X=vpm_out, meta=state.meta, phi=phi_new, goal=state.goal)

        delta = float(new_state.utility - state.utility)
        cost = float(thought.cost if thought.cost > 0 else self._estimate_cost(thought))
        bcs = max(delta - self.cfg.lambda_cost * cost, 0.0)
        return new_state, delta, cost, bcs

    # ----------------------------- internals -----------------------------

    def _apply_op(self, vpm_f: np.ndarray, op: VisualThoughtOp) -> np.ndarray:
        C, H, W = vpm_f.shape
        t = op.type

        if t is VisualThoughtType.ZOOM:
            cx, cy = _coerce_xy(op.params.get("center", (W // 2, H // 2)))
            scale = float(op.params.get("scale", 2.0))
            return _apply_zoom(vpm_f, (int(cx), int(cy)), max(1.0, float(scale)))

        elif t is VisualThoughtType.BBOX:
            x1, y1, x2, y2 = _coerce_xyxy(op.params.get("xyxy", (W // 4, H // 4, 3 * W // 4, 3 * H // 4)))
            thickness = int(op.params.get("thickness", 2))
            boost = float(op.params.get("boost", 0.15))
            _apply_bbox(vpm_f, (x1, y1, x2, y2), thickness=thickness, boost=boost)
            return vpm_f

        elif t is VisualThoughtType.PATH:
            pts = _coerce_points(op.params.get("points", [(W // 4, H // 2), (3 * W // 4, H // 2)]))
            thickness = int(op.params.get("thickness", 2))
            _apply_path(vpm_f, pts, arrows=bool(op.params.get("arrows", False)), thickness=thickness)
            return vpm_f

        elif t is VisualThoughtType.HIGHLIGHT:
            poly = _coerce_points(op.params.get("polygon", [(W // 3, H // 3), (2 * W // 3, H // 3), (2 * W // 3, 2 * H // 3), (W // 3, 2 * H // 3)]))
            opacity = float(op.params.get("opacity", 0.25))
            boost = float(op.params.get("boost", 0.25))
            _apply_highlight(vpm_f, poly, opacity=opacity, boost=boost)
            return vpm_f

        elif t is VisualThoughtType.BLUR:
            x1, y1, x2, y2 = _coerce_xyxy(op.params.get("xyxy", (W // 4, H // 4, 3 * W // 4, 3 * H // 4)))
            k = int(op.params.get("k", 3))
            passes = int(op.params.get("passes", 1))
            _apply_blur(vpm_f, (x1, y1, x2, y2), k=max(1, k), passes=max(1, passes))
            return vpm_f

        # Unknown op → no-op
        return vpm_f

    def _estimate_cost(self, thought: Thought) -> float:
        total = 0.0
        for op in thought.ops:
            total += float(self.cfg.visual_op_cost.get(op.type.value, 0.5))
        return total


# --------------------------------------------------------------------------------------
# Param coercion & validation
# --------------------------------------------------------------------------------------

def _coerce_xy(val: Any) -> Tuple[int, int]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return int(val[0]), int(val[1])
    return 0, 0


def _coerce_xyxy(val: Any) -> Tuple[int, int, int, int]:
    if isinstance(val, (list, tuple)) and len(val) == 4:
        return int(val[0]), int(val[1]), int(val[2]), int(val[3])
    return 0, 0, 0, 0


def _coerce_points(val: Any) -> List[Tuple[int, int]]:
    if isinstance(val, (list, tuple)):
        out: List[Tuple[int, int]] = []
        for p in val:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                out.append((int(p[0]), int(p[1])))
        if out:
            return out
    return []


# --------------------------------------------------------------------------------------
# Minimal doctest
# --------------------------------------------------------------------------------------

if __name__ == "__main__":  # quick sanity
    H = W = 64
    X = np.zeros((3, H, W), dtype=np.uint8)
    state = VPMState(X=X, meta={"positions": {}}, phi={"separability": 0.0}, goal=VPMGoal(weights={"separability": 1.0}))
    ex = ThoughtExecutor()

    th = Thought(
        name="ZoomBridge",
        ops=[
            VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (32, 32), "scale": 2.0}),
            VisualThoughtOp(VisualThoughtType.BBOX, {"xyxy": (16, 16, 48, 48), "thickness": 2, "boost": 0.2}),
        ],
        intent="Focus and highlight central region",
    )
    new_state, delta, cost, bcs = ex.score_thought(state, th)
    print("delta:", delta, "cost:", cost, "bcs:", bcs, "utility:", new_state.utility)
