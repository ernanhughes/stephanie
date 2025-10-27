# stephanie/scoring/scorer/vpm_transformer_scorer.py
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.model.vpm_model import (AttentionMap,
                                               TinyVisionTransformer,
                                               VPMDimension)
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer

log = logging.getLogger(__name__)

SUPPORTED = [
    VPMDimension.CLARITY,
    VPMDimension.NOVELTY,
    VPMDimension.CONFIDENCE,
    VPMDimension.CONTRADICTION,
    VPMDimension.COHERENCE,
    VPMDimension.COMPLEXITY,
    VPMDimension.ALIGNMENT,
]

class VPMScorer(BaseScorer):
    def __init__(self, cfg: Dict[str, Any], memory, container, logger=None):
        super().__init__(cfg, memory, container, logger or log)

        self.img_size   = cfg.get("img_size", 64)
        self.patch_size = cfg.get("patch_size", 8)
        self.in_channels= cfg.get("in_channels", 3)
        self.embed_dim  = cfg.get("embed_dim", 128)
        self.depth      = cfg.get("depth", 4)
        self.num_heads  = cfg.get("num_heads", 8)
        self.dropout    = cfg.get("dropout", 0.1)
        self.mlp_ratio  = cfg.get("mlp_ratio", 4.0)
        self.weights    = cfg.get("weights_path", None)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = TinyVisionTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            num_dimensions=len(SUPPORTED),
        ).to(self.device)

        if self.weights:
            try:
                state = torch.load(self.weights, map_location="cpu")
                self.model.load_state_dict(state)
                self.logger.info(f"Loaded VPM weights from {self.weights}")
            except Exception as e:
                self.logger.warning(f"Could not load VPM weights: {e}")

        self.model.eval()

    # ---------- public API ----------
    def _score_core(self, context: dict, scorable: Scorable, dimensions: List[str]) -> ScoreBundle:
        try:
            img = self._to_tensor(scorable).to(self.device)  # (1,C,H,W) in [0,1]
            with torch.no_grad():
                out = self.model(img, return_attention=False)
                raw = out["scores"].squeeze(0).detach().cpu().numpy()  # (D,)

            # Map model outputs to named dims
            base = {dim: float(raw[i]) for i, dim in enumerate(SUPPORTED)}

            # Apply importance/order if provided
            w = (scorable.meta or {}).get("dimension_weights", {})
            order = (scorable.meta or {}).get("dimension_order", [])
            weighted = self._apply_importance(base, w, order)

            # Build results for requested dims
            results: Dict[str, ScoreResult] = {}
            for dim in dimensions:
                if dim in base:
                    results[dim] = ScoreResult(
                        dimension=dim,
                        score=weighted[dim],
                        rationale=self._rationale(dim, weighted[dim]),
                        source="vpm_transformer",
                        attributes={"raw_score": base[dim], "weight": w.get(dim, 1.0)}
                    )
            # Always emit a composite if requested or if weights/order exist
            if "vpm_overall" in dimensions or w or order:
                comp = self._composite(weighted, w, order)
                results["vpm_overall"] = ScoreResult(
                    dimension="vpm_overall",
                    score=comp,
                    rationale=f"Weighted composite honoring order {order or 'none'}",
                    source="vpm_transformer",
                    meta={"dimension_weights": w, "dimension_order": order}
                )

            return ScoreBundle(results=results)

        except Exception as e:
            self.logger.error(f"VPM scoring failed: {e}")
            return self._fallback(dimensions)

    # ---------- helpers ----------
    def _to_tensor(self, scorable: Scorable) -> torch.Tensor:
        arr = scorable.get_image_array()
        if isinstance(arr, list):
            arr = np.array(arr)

        # Accept HxW, HxWx1, HxWx3, uint8 or float
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        # Binary friendly: keep exact 0/1 if already binary
        uniq = np.unique(arr)
        if uniq.size <= 3 and set(np.round(uniq, 3)).issubset({0.0, 1.0}):
            arr = (arr > 0.5).astype(np.float32)

        # Up/Downscale to model size
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        if x.shape[-2:] != (self.img_size, self.img_size):
            mode = (scorable.meta or {}).get("resize_method", "bilinear")
            x = torch.nn.functional.interpolate(x, size=(self.img_size, self.img_size),
                                                mode=mode, align_corners=False if mode=="bilinear" else None)
        # Adapt channels (binary maps often 1-channel)
        if x.shape[1] == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] != self.in_channels:
            # Last resort: project to expected channels
            if self.in_channels == 1:
                x = x.mean(dim=1, keepdim=True)
            else:
                x = x.repeat(1, self.in_channels // x.shape[1], 1, 1)[:, :self.in_channels]

        return x.clamp(0, 1)

    def _apply_importance(self, base: Dict[str, float], weights: Dict[str, float], order: List[str]) -> Dict[str, float]:
        """Apply per-dimension weights and (optional) order decay."""
        if not weights and not order:
            return base

        # Order decay: earlier dims in order list get multiplicative bonus
        decay = {}
        if order:
            # geometric decay (e.g., 1.0, 0.9, 0.81, ...)
            gamma = 0.9
            for i, d in enumerate(order):
                decay[d] = gamma ** i

        adjusted = {}
        for d, s in base.items():
            w = weights.get(d, 1.0) * decay.get(d, 1.0)
            adjusted[d] = float(np.clip(s * w, 0.0, 1.0))
        return adjusted

    def _composite(self, adjusted: Dict[str, float], weights: Dict[str, float], order: List[str]) -> float:
        if not adjusted:
            return 0.5
        # Normalize weights (after order decay)
        w = {d: weights.get(d, 1.0) for d in adjusted}
        if order:
            gamma = 0.9
            for i, d in enumerate(order):
                if d in w:
                    w[d] *= gamma ** i
        total = sum(w.values()) or 1.0
        return float(np.clip(sum(adjusted[d] * w[d] for d in adjusted) / total, 0.0, 1.0))

    def _rationale(self, dim: str, score: float) -> str:
        if dim == "clarity":
            return ("High" if score > 0.8 else "Moderate" if score > 0.6 else "Low") + " clarity pattern."
        if dim == "novelty":
            return ("Strongly novel" if score > 0.8 else "Some novelty" if score > 0.6 else "Familiar pattern.")
        if dim == "confidence":
            return ("Confident signal" if score > 0.8 else "Mixed certainty" if score > 0.6 else "Weak signal.")
        if dim == "contradiction":
            return ("Conflicts present" if score > 0.8 else "Some conflicts" if score > 0.6 else "Consistent signal.")
        return f"{dim.capitalize()} estimated."

    def _fallback(self, dimensions: List[str]) -> ScoreBundle:
        results = {d: ScoreResult(d, 0.5, "Fallback VPM score.", "vpm_transformer_fallback") for d in dimensions}
        return ScoreBundle(results=results)


def create_vpm_transformer_scorer(cfg, memory, container, logger):
    return VPMScorer(cfg, memory, container, logger)
