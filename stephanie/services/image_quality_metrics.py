from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import math
import numpy as np

try:
    from PIL import Image, ImageFilter, ImageStat
except Exception:
    Image = None  # type: ignore

class ImageQualityMetrics:
    """
    Lightweight image metrics with only Pillow + numpy.
    All scores normalized to [0,1] where higher is better (except noise_level).
    If PIL isn't available, returns neutral metrics.
    """

    def __init__(self, image_path: Optional[str] = None):
        self._last_img = None
        if image_path and Image:
            self._last_img = Image.open(image_path).convert("RGB")

    def get_metrics(self, image: Optional["Image.Image"] = None) -> Dict[str, float]:
        if Image is None or (image is None and self._last_img is None):
            # neutral fallbacks
            return {
                "sharpness": 0.5,
                "color_diversity": 0.5,
                "composition": 0.5,
                "aesthetic_score": 0.5,
                "relevance": 0.5,      # filled by CLIP layer if available
                "noise_level": 0.5,    # lower is better
                "contrast": 0.5,
                "color_balance": 0.5
            }

        img = image or self._last_img
        # work on smaller view for fast metrics
        small = img.resize((384, 384))

        sharpness = self._sharpness(small)
        color_div = self._color_diversity(small)
        composition = self._composition_rule_of_thirds(small)
        contrast = self._contrast(small)
        color_balance = self._color_balance(small)
        noise = self._noise_level(small)

        # A simple aesthetic proxy (blend of the above)
        aesthetic = 0.25 * sharpness + 0.20 * color_div + 0.20 * composition + 0.20 * contrast + 0.15 * color_balance

        return {
            "sharpness": float(sharpness),
            "color_diversity": float(color_div),
            "composition": float(composition),
            "aesthetic_score": float(aesthetic),
            "relevance": 0.5,  # set by CLIP layer (if used)
            "noise_level": float(noise),
            "contrast": float(contrast),
            "color_balance": float(color_balance),
        }

    # ---------- metric helpers ----------
    def _np(self, pil_img: "Image.Image") -> np.ndarray:
        return np.asarray(pil_img).astype(np.float32) / 255.0

    def _sharpness(self, img: "Image.Image") -> float:
        # Laplacian variance proxy via PIL edge-enhance diff
        edges = img.filter(ImageFilter.FIND_EDGES)
        stat = ImageStat.Stat(edges.convert("L"))
        # normalize by a rough max observed variance
        var = stat.var[0]
        return max(0.0, min(1.0, var / 8000.0))

    def _color_diversity(self, img: "Image.Image") -> float:
        # Histogram entropy across RGB
        arr = self._np(img)
        hist, _ = np.histogram(arr.reshape(-1, 3), bins=32, range=(0,1))
        p = hist / max(1, hist.sum())
        entropy = -(p[p>0] * np.log2(p[p>0])).sum()
        # normalize: max entropy ~ log2(32) ≈ 5
        return max(0.0, min(1.0, entropy / 5.0))

    def _contrast(self, img: "Image.Image") -> float:
        gray = img.convert("L")
        arr = np.asarray(gray).astype(np.float32) / 255.0
        std = float(arr.std())
        # heuristic normalize
        return max(0.0, min(1.0, std / 0.25))

    def _color_balance(self, img: "Image.Image") -> float:
        # distance of per-channel means from 0.5 (mid), lower distance = better balance
        arr = self._np(img)
        means = arr.reshape(-1, 3).mean(axis=0)
        dist = float(np.abs(means - 0.5).mean())  # 0..0.5
        # invert and normalize
        return max(0.0, min(1.0, 1.0 - (dist / 0.5)))

    def _noise_level(self, img: "Image.Image") -> float:
        # high-frequency energy proxy (laplacian-like) → normalize to [0,1] (higher = noisier)
        edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
        arr = np.asarray(edges).astype(np.float32) / 255.0
        e = float(arr.mean())
        return max(0.0, min(1.0, e))  # in [0,1]

    def _composition_rule_of_thirds(self, img: "Image.Image") -> float:
        # approximate composition by edge density near thirds intersections
        w, h = img.size
        thirds = [(w//3, h//3), (2*w//3, h//3), (w//3, 2*h//3), (2*w//3, 2*h//3)]
        edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
        arr = np.asarray(edges).astype(np.float32) / 255.0
        H, W = arr.shape
        score = 0.0
        for (x, y) in thirds:
            x0, x1 = max(0, x-16), min(W, x+16)
            y0, y1 = max(0, y-16), min(H, y+16)
            patch = arr[y0:y1, x0:x1]
            score += float(patch.mean())
        score /= len(thirds)
        # normalize; more edges near thirds → higher score
        return max(0.0, min(1.0, score))
