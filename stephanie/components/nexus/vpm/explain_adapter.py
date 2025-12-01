from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from zeromodel.vpm.explain import OcclusionVPMInterpreter


def occlusion_importance_from_vpm(
    vpm_uint8: np.ndarray,
    *,
    patch_h: int = 12,
    patch_w: int = 12,
    stride: int = 8,
    prior: str = "top_left",
    channel_agg: str = "mean",
) -> Tuple[np.ndarray, Dict]:
    """
    Gradient-free occlusion importance directly on an RGB VPM frame (H,W,3) uint8 or (H,W,C) float.
    Returns (importance01 [H,W], meta).
    """
    v = vpm_uint8
    if v.ndim == 3 and v.dtype != np.uint8:
        v = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    if v.ndim == 2:
        v = np.repeat(v[:, :, None], 3, axis=2)

    H, W, _ = v.shape
    # Positional weights (top-left bias)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.sqrt(yy**2 + xx**2)
    w = 1.0 - 0.3 * dist if prior == "top_left" else np.ones_like(dist)
    if w.max() > 0: w /= w.max()

    # Luminance
    v01 = v.astype(np.float32) / 255.0
    lum = v01.max(axis=2) if channel_agg == "max" else v01.mean(axis=2)

    # Base score = weighted luminance mean
    denom = float(w.sum()) + 1e-12
    base_score = float((lum * w).sum() / denom)

    # Zero baseline
    baseline = np.zeros_like(v01, dtype=np.float32)
    imp = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(H, y + patch_h); x2 = min(W, x + patch_w)
            patched = v01.copy()
            patched[y:y2, x:x2, :] = baseline[y:y2, x:x2, :]
            pl = patched.max(axis=2) if channel_agg == "max" else patched.mean(axis=2)
            occ_score = float((pl * w).sum() / denom)
            drop = max(0.0, base_score - occ_score)
            imp[y:y2, x:x2] += drop

    if imp.max() > 0: imp /= imp.max()
    meta = dict(base_score=base_score, patch_h=patch_h, patch_w=patch_w, stride=stride, prior=prior, channel_agg=channel_agg)
    return imp.astype(np.float32), meta


def occlusion_importance_from_zeromodel(
    zeromodel, *,  # ZeroModel with sorted_matrix prepared
    patch_h: int = 12, patch_w: int = 12, stride: int = 8, prior: str = "top_left", channel_agg: str = "mean"
) -> Tuple[np.ndarray, Dict]:
    """
    If your ZeroModelService exposes a prepared ZeroModel with sorted_matrix,
    use the provided interpreter; else fall back to the VPM-based path.
    """
    interpreter = OcclusionVPMInterpreter(
        patch_h=patch_h, patch_w=patch_w, stride=stride,
        prior=prior, channel_agg=channel_agg
    )
    imp, meta = interpreter.explain(zeromodel)
    return imp.astype(np.float32), meta
