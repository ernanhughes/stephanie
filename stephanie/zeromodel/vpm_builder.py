# stephanie/zeromodel/vpm_builder.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageOps  # pillow


@dataclass
class VPMConfig:
    metrics: List[str] = None
    image_scale: int = 2  # upsample for visibility

class CaseBookVPMBuilder:
    def __init__(self, tokenizer, metrics: List[str] = None, cfg: VPMConfig | None = None):
        self.tokenizer = tokenizer
        self.metrics = metrics or ["sicql", "ebt", "llm"]
        self.cfg = cfg or VPMConfig(metrics=self.metrics)

    def build(self, casebook, model) -> np.ndarray:
        """Return 2D float image of shape (num_cases, num_metrics) in [0,1]."""
        rows = []
        for case in casebook.cases:
            rows.append(self._scores_for_case(case, model))
        arr = np.array(rows, dtype=np.float32)
        return self._normalize(arr)

    def build_subset(self, cases, model) -> np.ndarray:
        rows = []
        for case in cases:
            rows.append(self._scores_for_case(case, model))
        arr = np.array(rows, dtype=np.float32)
        return self._normalize(arr)

    def _scores_for_case(self, case, model) -> List[float]:
        # Replace with real calls into your scoring stack
        # Placeholder: produce [sicql, ebt, llm] in [0,1]
        sicql = getattr(case, "sicql_score", 0.5)
        ebt   = getattr(case, "ebt_score", 0.5)
        llm   = getattr(case, "llm_score", 0.5)
        return [float(sicql), float(ebt), float(llm)]

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        mn, mx = float(arr.min()), float(arr.max())
        if mx <= mn + 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - mn) / (mx - mn + 1e-9)

    def save_image(self, vpm: np.ndarray, path: str, title: str | None = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Scale and convert to image
        vpm_uint8 = (np.clip(vpm, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(vpm_uint8, mode="L")
        if self.cfg.image_scale != 1:
            img = img.resize((img.width * self.cfg.image_scale, img.height * self.cfg.image_scale), Image.NEAREST)
        if title:
            img = ImageOps.expand(img, border=20, fill="white")
            draw = ImageDraw.Draw(img)
            draw.text((10, 5), title, fill=0)
        img.save(path)
