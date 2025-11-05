from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stephanie.scoring.scorable import Scorable

IMG_SIZE = 256

def _text_to_vpm(text: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Render a lightweight Text-VPM: 3 channels (token density, line breaks, emphasis)
    - C0: character density heat
    - C1: line-break map
    - C2: simple emphasis proxy (uppercase & punctuation density)
    Returns uint8 array [3, H, W] in [0,255].
    """
    H = W = img_size
    heat = np.zeros((H, W), dtype=np.float32)
    breaks = np.zeros_like(heat)
    emph = np.zeros_like(heat)

    # coarse raster: map characters into a grid
    # grid ~ 64x64, then upscale
    gh = gw = 64
    cell_h = H // gh
    cell_w = W // gw

    lines = text.splitlines()[:512] or [text[:2048]]
    for li, line in enumerate(lines[:gh]):
        y = min(gh - 1, li)
        # density by slice
        dens = min(1.0, len(line) / 200.0)
        xcells = min(gw, max(1, len(line) // 4))
        for xi in range(xcells):
            heat[y*cell_h:(y+1)*cell_h, xi*cell_w:(xi+1)*cell_w] += dens
        breaks[y*cell_h:(y+1)*cell_h, :] += 1.0

        # emphasis: uppercase and punctuation ratio
        if line:
            caps = sum(1 for c in line if c.isupper())
            punc = sum(1 for c in line if c in "!?;:.")
            ratio = min(1.0, 0.5*caps/len(line) + 0.5*punc/len(line))
            emph[y*cell_h:(y+1)*cell_h, : int(ratio * W)] += ratio

    # normalize each to [0,255]
    def norm_u8(a):
        a = a - a.min()
        m = a.max() or 1.0
        return (255.0 * (a / m)).astype(np.uint8)

    return np.stack([norm_u8(heat), norm_u8(breaks), norm_u8(emph)], axis=0)

def scorable_to_vpm(s: Scorable, out_dir: Path) -> Tuple[np.ndarray, Dict]:
    """
    Returns (vpm_uint8 [3,H,W], meta).
    If scorable.meta['vpm_path'] exists, load it.
    Else if scorable.meta['image_path'] exists, load and normalize to 3-ch gray stack.
    Else synthesize a Text-VPM from scorable.text.
    """
    meta = {"source": s.target_type, "scorable_id": s.id}
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Existing VPM on disk
    vpm_path = (s.meta or {}).get("vpm_path")
    if vpm_path:
        arr = np.array(Image.open(vpm_path).convert("L"), dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=0)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr = np.transpose(arr, (2,0,1))
        meta["adapter"] = "existing_vpm_path"
        return arr, meta

    # 2) Raw image path → make 3-ch grayscale stack as VPM
    img_path = (s.meta or {}).get("image_path")
    if img_path:
        im = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        g = np.array(im, dtype=np.uint8)
        arr = np.stack([g, g, g], axis=0)
        meta["adapter"] = "image_to_vpm_gray3"
        return arr, meta

    # 3) Text → Text-VPM
    txt = s.text or ""
    arr = _text_to_vpm(txt, IMG_SIZE)
    meta["adapter"] = "text_to_vpm"
    meta["text_len"] = len(txt)

    # dump the preview PNG for reports
    png = np.transpose(arr, (1,2,0))  # HWC
    Image.fromarray(png).save(out_dir / f"scorable_{hashlib.md5((s.id+txt[:64]).encode()).hexdigest()[:8]}.png")
    return arr, meta
