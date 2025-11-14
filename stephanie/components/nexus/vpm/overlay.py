# stephanie/components/nexus/vpm/overlay.py
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def overlay_heat(base_hwc_u8: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """
    base: [H,W,C] uint8, heat: [H,W] float in [0,1]
    Returns HWC uint8 with a soft red overlay.
    """
    H, W, C = base_hwc_u8.shape
    heat_rgb = np.zeros((H, W, 3), dtype=np.float32)
    heat_rgb[..., 0] = heat  # red
    out = base_hwc_u8.astype(np.float32)/255.0
    out = np.clip(out * 0.8 + heat_rgb * 0.6, 0.0, 1.0)
    return (out * 255).astype(np.uint8)
