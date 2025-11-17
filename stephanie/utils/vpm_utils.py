# stephanie/utils/vpm_utils.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)

# ---------------------------
# Shape & normalization core
# ---------------------------

def detect_vpm_layout(a: Any) -> str:
    """
    Return "CHW" | "HWC" | "HW" | "1D" for common cases.
    """
    arr = np.asarray(a)
    if arr.ndim == 1:
        return "1D"
    if arr.ndim == 2:
        return "HW"
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):
            return "CHW"
        if arr.shape[-1] in (1, 3, 4):
            return "HWC"
    return f"UNK{arr.shape}"

def _to_float32(a: Any) -> np.ndarray:
    x = np.asarray(a)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def _normalize_to_01(img: np.ndarray) -> np.ndarray:
    """
    Heuristic normalization to [0,1] (handles [0,1], ~[-1,1], or 0..255-ish).
    Works for any shape; returns float32 in [0,1].
    """
    x = _to_float32(img)
    if x.size == 0:
        return x
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    # Already 0..1?
    if 0.0 <= mn and mx <= 1.0:
        return np.clip(x, 0.0, 1.0)
    # ~[-1,1]?
    if -1.1 <= mn <= 0.0 <= mx <= 1.1:
        return np.clip((x - mn) / (mx - mn), 0.0, 1.0)
    # Otherwise assume 0..255 or wider
    return np.clip(x / 255.0, 0.0, 1.0)

def ensure_hwc_u8(vpm: Any, *, force_three: bool = True) -> np.ndarray:
    """
    Accept 1D, HW, HWC, or CHW; return HWC uint8. If single channel and force_three,
    tile to 3 channels.
    """
    arr = np.asarray(vpm)
    lay = detect_vpm_layout(arr)

    # Canonicalize to HWC float [0,1] first
    if lay == "1D":
        arr = arr[None, :]               # 1 x D "row image"
        lay = "HW"
    if lay == "HW":
        arr = arr[..., None]             # H x W x 1
        lay = "HWC"
    if lay == "CHW":
        arr = np.transpose(arr, (1, 2, 0))  # HWC
        lay = "HWC"

    if lay != "HWC":
        raise ValueError(f"ensure_hwc_u8: unsupported layout {lay}")

    # Normalize to [0,1], then to 0..255 u8
    x01 = _normalize_to_01(arr)
    u8 = (x01 * 255.0).round().clip(0, 255).astype(np.uint8)

    # Enforce channels
    c = u8.shape[-1]
    if c == 1 and force_three:
        u8 = np.repeat(u8, 3, axis=-1)
    elif c > 3:
        u8 = u8[..., :3]
    return u8

def ensure_chw_u8(vpm: Any, *, force_three: bool = True) -> np.ndarray:
    """
    Accept anything; return CHW uint8 with C in {1,3}.
    """
    hwc = ensure_hwc_u8(vpm, force_three=force_three)
    return np.transpose(hwc, (2, 0, 1))

def composite_gray_u8(vpm: Any) -> np.ndarray:
    """
    Convert any layout to grayscale HxW uint8 by averaging channels (after normalization).
    """
    hwc = ensure_hwc_u8(vpm, force_three=True)
    gray = hwc.mean(axis=-1)
    return gray.astype(np.uint8, copy=False)

# ---------------------------
# Stats & persistence
# ---------------------------

def image_stats(img: Any) -> Dict[str, float]:
    x = _to_float32(img)
    if x.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p1": 0.0, "p99": 0.0}
    return {
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "p1": float(np.nanpercentile(x, 1)),
        "p99": float(np.nanpercentile(x, 99)),
    }

def save_png_u8(img_u8_hw: np.ndarray, path: Path) -> Optional[str]:
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(img_u8_hw), mode="L").save(path)
        return str(path)
    except Exception as e:
        log.warning(f"save_png_u8 failed at {path}: {e}")
        return None

def vpm_image_details(
    vpm: Any,
    out_path: Path,
    *,
    save_per_channel: bool = False,
) -> Dict[str, Any]:
    """
    Normalize any CHW/HWC/HW/1D VPM to grayscale uint8, save it,
    optionally save per-channel PNGs, and return descriptive stats & file paths.
    """
    arr = np.asarray(vpm)
    layout = detect_vpm_layout(arr)
    shape = tuple(arr.shape)
    dtype = str(arr.dtype)

    # Compose grayscale + u8
    gray_u8 = composite_gray_u8(arr)

    # Stats (pre-normalization, informative)
    stats = image_stats(arr)

    # Degenerate if zero range or non-finite spread
    is_deg = (stats["max"] <= stats["min"]) or (not np.isfinite(stats["max"]) or not np.isfinite(stats["min"]))
    note = "degenerate composite (non-finite or zero range)" if is_deg else "ok"

    # Save grayscale
    out_path = Path(out_path)
    gray_path = save_png_u8(gray_u8, out_path)

    # Optionally save per-channel
    channel_paths: List[str] = []
    if save_per_channel:
        # Rebuild HWC u8 (3ch) and dump L per channel
        hwc_u8 = ensure_hwc_u8(arr, force_three=True)
        for idx in range(hwc_u8.shape[-1]):
            ch = hwc_u8[..., idx]
            ch_path = out_path.with_name(f"{out_path.stem}_ch{idx}.png")
            p = save_png_u8(ch, ch_path)
            if p:
                channel_paths.append(p)

    log.info(
        "VPM details: layout=%s shape=%s dtype=%s min=%.4f max=%.4f mean=%.4f std=%.4f p1=%.4f p99=%.4f "
        "degenerate=%s gray_saved=%s channels_saved=%d",
        layout, shape, dtype, stats["min"], stats["max"], stats["mean"], stats["std"], stats["p1"], stats["p99"],
        bool(is_deg), bool(gray_path), len(channel_paths)
    )

    return {
        "gray_path": gray_path,
        "channel_paths": channel_paths,
        "shape": shape,
        "dtype": dtype,
        "layout": layout,
        "stats": stats,
        "is_degenerate": bool(is_deg),
        "note": note,
    }

# ---------------------------
# Quick debug dump (optional)
# ---------------------------

def vpm_quick_dump(vpm: Any, out_dir: Path, base: str = "vpm") -> Dict[str, Any]:
    """
    Save gray + channels + info JSON to out_dir/base.*  Returns paths & metadata.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gray_png = out_dir / f"{base}.png"
    info = vpm_image_details(vpm, gray_png, save_per_channel=True)
    info_path = out_dir / f"{base}.json"
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    except Exception as e:
        log.warning(f"vpm_quick_dump could not write info JSON: {e}")
    info["json_path"] = str(info_path)
    return info


def to_rgb(u8_chw: np.ndarray) -> np.ndarray:
    """[C,H,W] (uint8/float) → [H,W,3] uint8 for visualization."""
    X = np.asarray(u8_chw)
    assert X.ndim == 3, f"expected [C,H,W], got {X.shape}"
    C, H, W = X.shape
    if C == 3:
        rgb = np.transpose(X, (1, 2, 0))
    elif C == 1:
        ch = np.transpose(X, (1, 2, 0))
        rgb = np.repeat(ch, 3, axis=2)
    else:
        # take first 3 channels if available; else tile the first
        if C >= 3:
            rgb = np.transpose(X[:3], (1, 2, 0))
        else:
            ch = np.transpose(X[:1], (1, 2, 0))
            rgb = np.repeat(ch, 3, axis=2)
    # map to uint8 safely
    if rgb.dtype != np.uint8:
        # assume in [0,1] or arbitrary range; clip & scale
        rmin, rmax = float(np.min(rgb)), float(np.max(rgb))
        if rmax > 1.0 or rmin < 0.0:
            # min-max normalize
            denom = (rmax - rmin) if (rmax - rmin) > 1e-9 else 1.0
            rgb = (np.clip((rgb - rmin) / denom, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb

def mass(x01: np.ndarray) -> float:
    """Average intensity (proxy for coverage/strength)."""
    x = np.asarray(x01, dtype=np.float32)
    if x.ndim > 2:
        x = x.mean(axis=0)
    return float(np.mean(np.clip(x, 0.0, 1.0)))


def alignment(a01: np.ndarray, b01: np.ndarray) -> float:
    """Cosine-like alignment between two 2D maps in [0,1]."""
    A = np.asarray(a01, dtype=np.float32).ravel()
    B = np.asarray(b01, dtype=np.float32).ravel()
    num = float((A * B).sum())
    den = float(np.linalg.norm(A) * np.linalg.norm(B)) + 1e-9
    return num / den

def occlusion_importance(
    vpm_rgb_u8: np.ndarray,
    *,
    patch_h: int = 12,
    patch_w: int = 12,
    stride: int = 8,
    prior: str = "top_left",
    channel_agg: str = "mean",
) -> np.ndarray:
    """
    Gradient-free occlusion importance directly on the VPM RGB image.
    Returns [H,W] float in [0,1].
    """
    v = vpm_rgb_u8
    assert v.ndim == 3 and v.shape[2] == 3, f"expected RGB [H,W,3], got {v.shape}"
    H, W, _ = v.shape

    # positional weights
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if prior == "top_left":
        dist = np.sqrt(yy**2 + xx**2)
        w = 1.0 - 0.3 * dist
        w[w < 0] = 0.0
        if w.max() > 0:
            w /= w.max()
    else:
        w = np.ones((H, W), dtype=np.float32)

    v01 = v.astype(np.float32) / 255.0
    lum = v01.max(axis=2) if channel_agg == "max" else v01.mean(axis=2)
    denom = float(w.sum()) + 1e-12
    base = float((lum * w).sum() / denom)

    # zero baseline
    imp = np.zeros((H, W), dtype=np.float32)
    baseline = np.zeros_like(v01, dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(H, y + patch_h)
            x2 = min(W, x + patch_w)
            patched = v01.copy()
            patched[y:y2, x:x2, :] = baseline[y:y2, x:x2, :]
            pl = patched.max(axis=2) if channel_agg == "max" else patched.mean(axis=2)
            occ = float((pl * w).sum() / denom)
            drop = max(0.0, base - occ)
            imp[y:y2, x:x2] += drop

    if imp.max() > 0:
        imp /= imp.max()
    return imp.astype(np.float32)


def resize_bilinear_np(X01: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    # X01: [C,H,W] float32 in [0,1]
    C, H, W = X01.shape
    if H == new_h and W == new_w:
        return X01.copy() 
    ys = np.linspace(0, H - 1, new_h, dtype=np.float32)
    xs = np.linspace(0, W - 1, new_w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32); y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.floor(xs).astype(np.int32); x1 = np.clip(x0 + 1, 0, W - 1)
    wy = (ys - y0).reshape(1, new_h, 1)      # [1,Nh,1]
    wx = (xs - x0).reshape(1, 1, new_w)      # [1,1,Nw]

    out = np.empty((C, new_h, new_w), dtype=np.float32)
    for c in range(C):
        Ia = X01[c][y0[:, None], x0[None, :]]
        Ib = X01[c][y1[:, None], x0[None, :]]
        Ic = X01[c][y0[:, None], x1[None, :]]
        Id = X01[c][y1[:, None], x1[None, :]]
        top = Ia * (1 - wx) + Ic * wx
        bot = Ib * (1 - wx) + Id * wx
        out[c] = top * (1 - wy) + bot * wy
    return np.clip(out, 0.0, 1.0)

def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if mx - mn <= 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)
