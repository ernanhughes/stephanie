# stephanie/utils/vpm_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import logging
log = logging.getLogger(__name__)


def vpm_image_details(
    vpm: Any,
    out_path: Path,
    *,
    save_per_channel: bool = False,
    logger: Any = None,
) -> Dict[str, Any]:
    """
    Normalize a CHW/HWC VPM array to a grayscale uint8 image, save it,
    optionally save per-channel PNGs, and return descriptive stats & file paths.

    Args:
        vpm:            2D or 3D array-like. If 3D, accepts CHW or HWC.
        out_path:       Where to save the grayscale composite ('.png' suggested).
        save_per_channel:
                        If True and vpm is multi-channel, saves vpm_ch{idx}.png alongside.
        logger:         Optional logger (must support .debug/.warning).

    Returns:
        dict with:
          - gray_path: str | None
          - channel_paths: List[str]
          - shape: tuple
          - dtype: str
          - layout: "CHW" | "HWC" | "HW"
          - stats: {min,max,mean,std,p1,p99}
          - is_degenerate: bool
          - note: str
    """
    
    arr = np.asarray(vpm)
    shape = tuple(arr.shape)
    dtype = str(arr.dtype)

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):  # CHW
            layout = "CHW"
            comp = arr.mean(axis=0)
            channels = [arr[c] for c in range(arr.shape[0])]
        elif arr.shape[-1] in (1, 3, 4):  # HWC
            layout = "HWC"
            comp = arr.mean(axis=-1)
            channels = [arr[..., c] for c in range(arr.shape[-1])]
        else:
            raise ValueError(f"Unexpected VPM shape {shape} (need CHW or HWC with C in (1,3,4)).")
    elif arr.ndim == 2:
        layout = "HW"
        comp = arr
        channels = [arr]
    else:
        raise ValueError(f"Unexpected VPM ndim={arr.ndim}; expected 2D/3D.")

    comp = comp.astype(np.float32)

    # Stats before normalization
    cmin = float(np.nanmin(comp)) if comp.size else 0.0
    cmax = float(np.nanmax(comp)) if comp.size else 0.0
    cmean = float(np.nanmean(comp)) if comp.size else 0.0
    cstd = float(np.nanstd(comp)) if comp.size else 0.0
    p1 = float(np.nanpercentile(comp, 1)) if comp.size else 0.0
    p99 = float(np.nanpercentile(comp, 99)) if comp.size else 0.0

    # Robust normalization to 0..255
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
        comp_u8 = np.zeros_like(comp, dtype=np.uint8)
        is_deg = True
        note = "degenerate composite (non-finite or zero range)"
    else:
        is_deg = False
        # Heuristics:
        # - [0,1] → scale *255
        # - [-1,1] (approx) → min-max stretch
        # - else assume already 0..255 and clip
        if 0.0 <= cmin and cmax <= 1.0:
            comp_u8 = (comp * 255.0).clip(0, 255).astype(np.uint8)
            note = "normalized from [0,1] → [0,255]"
        elif -1.1 <= cmin <= 0.0 <= cmax <= 1.1:
            comp_u8 = ((comp - cmin) / (cmax - cmin) * 255.0).clip(0, 255).astype(np.uint8)
            note = "min-max stretched from ~[-1,1] → [0,255]"
        else:
            comp_u8 = comp.clip(0, 255).astype(np.uint8)
            note = "assumed already 0..255; clipped+cast"

    # Save grayscale composite
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gray_path: Optional[str] = None
    try:
        Image.fromarray(np.ascontiguousarray(comp_u8), mode="L").save(out_path)
        gray_path = str(out_path)
    except Exception as e:
        log.warning(f"Failed to save grayscale VPM at {out_path}: {e}")

    # Optionally save per-channel PNGs
    channel_paths: List[str] = []
    if save_per_channel and len(channels) > 1:
        for idx, ch in enumerate(channels):
            try:
                ch = np.asarray(ch, dtype=np.float32)
                ch_min = float(np.nanmin(ch)) if ch.size else 0.0
                ch_max = float(np.nanmax(ch)) if ch.size else 0.0
                if not np.isfinite(ch_min) or not np.isfinite(ch_max) or ch_max <= ch_min:
                    ch_u8 = np.zeros_like(ch, dtype=np.uint8)
                elif 0.0 <= ch_min and ch_max <= 1.0:
                    ch_u8 = (ch * 255.0).clip(0, 255).astype(np.uint8)
                elif -1.1 <= ch_min <= 0.0 <= ch_max <= 1.1:
                    ch_u8 = ((ch - ch_min) / (ch_max - ch_min) * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    ch_u8 = ch.clip(0, 255).astype(np.uint8)

                ch_path = out_path.with_name(f"{out_path.stem}_ch{idx}.png")
                Image.fromarray(np.ascontiguousarray(ch_u8), mode="L").save(ch_path)
                channel_paths.append(str(ch_path))
            except Exception as e:
                log.warning(f"Failed to save channel {idx} for {out_path}: {e}")

    log.info(
        f"VPM image details: shape={shape} dtype={dtype} layout={layout} "
        f"min={cmin:.4f} max={cmax:.4f} mean={cmean:.4f} std={cstd:.4f} p1={p1:.4f} p99={p99:.4f} "
        f"degenerate={is_deg} saved_gray={bool(gray_path)} channels_saved={len(channel_paths)}"
    )

    return {
        "gray_path": gray_path,
        "channel_paths": channel_paths,
        "shape": shape,
        "dtype": dtype,
        "layout": layout,
        "stats": {
            "min": cmin,
            "max": cmax,
            "mean": cmean,
            "std": cstd,
            "p1": p1,
            "p99": p99,
        },
        "is_degenerate": bool(is_deg),
        "note": note,
    }
