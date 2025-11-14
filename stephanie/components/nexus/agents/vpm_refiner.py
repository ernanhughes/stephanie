# stephanie/components/nexus/agents/vpm_refiner.py
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import imageio.v2 as iio
import numpy as np
from PIL import Image

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.nexus.vpm.maps import MapProvider
from stephanie.components.nexus.vpm.state_machine import (ThoughtExecutor,
                                                          VPMGoal, VPMState,
                                                          compute_phi)
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_processor import ScorableProcessor
from stephanie.services.zeromodel_service import ZeroModelService

log = logging.getLogger(__name__)


# ------------------------------- Config --------------------------------------

@dataclass
class VPMRefinerConfig:
    mode: str = "filmstrip"     # "online" | "filmstrip"
    out_root: str = "runs/vpm"
    img_size: int = 256
    max_steps: int = 1          # we do a logic+importance pass; keep 1 by default
    phi_threshold: float = 0.90
    # occlusion params (for importance)
    occ_patch_h: int = 12
    occ_patch_w: int = 12
    occ_stride: int = 8
    occ_prior: str = "top_left"  # "top_left" | "uniform"
    occ_channel_agg: str = "mean"  # "mean" | "max"

    def __post_init__(self):
        # no-op; kept for parity with your earlier cfg pattern
        pass


# ----------------------- Small internal helpers ------------------------------

def _to_rgb(u8_chw: np.ndarray) -> np.ndarray:
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


def _heat_overlay(rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a heat map (single channel [H,W] in [0,1]) onto an RGB image."""
    H, W, _ = rgb.shape
    h = (np.clip(heat01, 0.0, 1.0) * 255).astype(np.uint8)
    h = h[:H, :W]
    # red channel heat
    h_rgb = np.stack([h, np.zeros_like(h), np.zeros_like(h)], axis=-1)
    out = (rgb.astype(np.float32) * (1 - alpha) + h_rgb.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _mass(x01: np.ndarray) -> float:
    """Average intensity (proxy for coverage/strength)."""
    x = np.asarray(x01, dtype=np.float32)
    if x.ndim > 2:
        x = x.mean(axis=0)
    return float(np.mean(np.clip(x, 0.0, 1.0)))


def _alignment(a01: np.ndarray, b01: np.ndarray) -> float:
    """Cosine-like alignment between two 2D maps in [0,1]."""
    A = np.asarray(a01, dtype=np.float32).ravel()
    B = np.asarray(b01, dtype=np.float32).ravel()
    num = float((A * B).sum())
    den = float(np.linalg.norm(A) * np.linalg.norm(B)) + 1e-9
    return num / den


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if mx - mn <= 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)


def _logic_not(x01: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.asarray(x01, dtype=np.float32), 0.0, 1.0)


def _logic_and(a01: np.ndarray, b01: np.ndarray) -> np.ndarray:
    # fuzzy AND = min
    return np.minimum(np.asarray(a01, dtype=np.float32), np.asarray(b01, dtype=np.float32))


def _logic_or(a01: np.ndarray, b01: np.ndarray) -> np.ndarray:
    # fuzzy OR = max
    return np.maximum(np.asarray(a01, dtype=np.float32), np.asarray(b01, dtype=np.float32))


def _occlusion_importance(
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


# ----------------------------- The Agent -------------------------------------

class VPMRefinerAgent(BaseAgent):
    """
    VPM Thought Refiner (logic + occlusion edition)

    - Builds VPM from a scorable via ZeroModelService
    - Derives/fetches maps: quality, novelty, uncertainty (+ occlusion importance)
    - Composes logic attention: (quality ∧ ¬uncertainty) ∨ (novelty ∧ ¬uncertainty)
    - Decision map: attention ∧ importance
    - Saves 4-panel film + metrics
    """

    name = "nexus_vpm_refiner"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.cfg = VPMRefinerConfig(
            mode=str(cfg.get("mode", "filmstrip")),
            out_root=str(cfg.get("out_root", "runs/vpm")),
            img_size=int(cfg.get("img_size", 256)),
            max_steps=int(cfg.get("max_steps", 1)),
            phi_threshold=float(cfg.get("phi_threshold", 0.90)),
            occ_patch_h=int(cfg.get("occ_patch_h", 12)),
            occ_patch_w=int(cfg.get("occ_patch_w", 12)),
            occ_stride=int(cfg.get("occ_stride", 8)),
            occ_prior=str(cfg.get("occ_prior", "top_left")),
            occ_channel_agg=str(cfg.get("occ_channel_agg", "mean")),
        )

        # Services & helpers
        self.zm: ZeroModelService = container.get("zeromodel")
        self.map_provider = MapProvider(self.zm)  # expected to have .build(X) → maps
        self.executor: ThoughtExecutor = ThoughtExecutor(
            visual_op_cost={"zoom": 1.0, "bbox": 0.3, "path": 0.4, "highlight": 0.5, "blur": 0.6}
        )
        self.scorable_processor = ScorableProcessor(
            cfg=cfg,
            memory=memory,
            container=container,
            logger=logger,
        )


    # ----------------------------- Run ---------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        scorables = list(context.get("scorables") or [])
        if not scorables:
            context[self.output_key] = {"status": "no_scorables"}
            return context

        seed = Scorable.from_dict(scorables[0])
        row = await self.scorable_processor.process(seed, context=context)

        metrics_values = row.get("metrics_values", [])
        metrics_columns = row.get("metrics_columns", [])
        # Prepare ZeroModel and VPM
        self.zm.initialize()
        vpm_u8, adapter_meta = await self.zm.vpm_from_scorable(seed, metrics_values=metrics_values, metrics_columns=metrics_columns)  # [C,H,W] uint8
        # Make initial state
        state = VPMState(
            X=vpm_u8,
            meta={"adapter": adapter_meta, "maps": {}},
            phi=compute_phi(vpm_u8, {}),
            goal=VPMGoal(weights={"separability": 1.0, "bridge_proxy": -0.5}),
        )

        # Build/fetch semantic maps
        maps = self._build_maps_safe(state.X)
        state.meta["maps"].update(maps)

        # Occlusion-based importance (always computed — stable & model-agnostic)
        v_rgb = _to_rgb(state.X)
        imp = _occlusion_importance(
            v_rgb,
            patch_h=self.cfg.occ_patch_h,
            patch_w=self.cfg.occ_patch_w,
            stride=self.cfg.occ_stride,
            prior=self.cfg.occ_prior,
            channel_agg=self.cfg.occ_channel_agg,
        )
        state.meta["maps"]["importance"] = imp  # [H,W] in [0,1]

        # Compose logic attention + decision
        attention = self._compose_attention(maps)            # [H,W] in [0,1]
        decision = _logic_and(attention, imp)                # [H,W] in [0,1]

        # Ensure at least 4 channels in state.X for storing outputs
        state.X = self._ensure_channels(state.X, min_channels=4)
        # Channel layout:
        # 0: attention, 1: anti-uncertainty (mask), 2: reserved, 3: decision
        state.X[0] = (attention * 255).astype(np.uint8)
        state.X[3] = (decision * 255).astype(np.uint8)

        # Recompute φ after logic composition (purely to update report; optional)
        state.phi = compute_phi(state.X, state.meta)

        # Film + metrics
        frames: List[np.ndarray] = []
        metrics: List[Dict[str, float]] = []

        # Single “step” panel (we keep loop structure for future iterative ops)
        for step in range(self.cfg.max_steps):
            rgb = _to_rgb(state.X)
            att01 = attention
            imp01 = imp
            dec01 = decision

            panel = self._four_panel(rgb, att01, imp01, dec01)
            frames.append(panel)

            metrics.append(
                {
                    "attention_mass": _mass(att01),
                    "importance_mass": _mass(imp01),
                    "decision_mass": _mass(dec01),
                    "att_imp_alignment": _alignment(att01, imp01),
                }
            )

        report = self._finalize_report(state, metrics)

        if self.cfg.mode == "filmstrip":
            run_id = f"vpm-{uuid.uuid4().hex[:8]}"
            run_dir = Path(self.cfg.out_root) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            for i, fr in enumerate(frames):
                Image.fromarray(fr).save(run_dir / f"frame_{i:02d}.png")
            # simple GIF
            iio.mimsave(run_dir / "filmstrip.gif", frames, fps=1, loop=0)
            (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context["vpm_artifacts"] = {"run_dir": str(run_dir)}

        context[self.output_key] = report
        return context

    # --------------------------- Map plumbing --------------------------------

    def _build_maps_safe(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Try MapProvider; if any map is missing, derive a simple fallback so
        downstream logic always has quality/novelty/uncertainty in [H,W] [0,1].
        """
        H, W = X.shape[1], X.shape[2]
        try:
            built = self.map_provider.build(X)  # could be dict or object with .maps
            maps = built.maps if hasattr(built, "maps") else dict(built)
        except Exception as e:
            log.warning(f"MapProvider.build failed: {e}")
            maps = {}

        # Fallbacks (very conservative & deterministic)
        # quality: normalized mean over channels
        if "quality" not in maps:
            q = np.clip(X.astype(np.float32).mean(axis=0) / 255.0, 0.0, 1.0)
            maps["quality"] = q
        # novelty: edge-ish proxy via local contrast on the mean channel
        if "novelty" not in maps:
            m = X.astype(np.float32).mean(axis=0) / 255.0
            # simple high-pass proxy: m - box_blur(m)
            k = max(1, min(H, W) // 64)
            if k > 1:
                kernel = np.ones((k, k), dtype=np.float32) / (k * k)
                # separable blur
                blur_h = np.apply_along_axis(lambda r: np.convolve(r, np.ones(k)/k, mode="same"), 1, m)
                blur = np.apply_along_axis(lambda c: np.convolve(c, np.ones(k)/k, mode="same"), 0, blur_h)
                nov = _normalize01(m - blur)
            else:
                nov = _normalize01(m)
            maps["novelty"] = nov
        # uncertainty: inverse of quality as a safe placeholder
        if "uncertainty" not in maps:
            maps["uncertainty"] = _logic_not(maps["quality"])

        return {k: np.asarray(v, dtype=np.float32) for k, v in maps.items()}

    def _compose_attention(self, maps: Dict[str, np.ndarray]) -> np.ndarray:
        """
        attention = (quality ∧ ¬uncertainty) ∨ (novelty ∧ ¬uncertainty)
        All maps are [H,W] in [0,1].
        """
        q = _normalize01(maps.get("quality", 0.0))
        n = _normalize01(maps.get("novelty", 0.0))
        u = _normalize01(maps.get("uncertainty", 1.0))
        not_u = _logic_not(u)
        return _logic_or(_logic_and(q, not_u), _logic_and(n, not_u))

    # ----------------------------- Viz & report -------------------------------

    def _four_panel(self, rgb: np.ndarray, att01: np.ndarray, imp01: np.ndarray, dec01: np.ndarray) -> np.ndarray:
        p1 = rgb
        p2 = _heat_overlay(rgb, att01)
        p3 = _heat_overlay(rgb, imp01)
        p4 = _heat_overlay(rgb, dec01)
        return np.concatenate([p1, p2, p3, p4], axis=1)

    def _ensure_channels(self, X: np.ndarray, min_channels: int) -> np.ndarray:
        X = np.asarray(X)
        C, H, W = X.shape
        if C >= min_channels:
            return X
        extra = np.zeros((min_channels - C, H, W), dtype=X.dtype)
        return np.concatenate([X, extra], axis=0)

    def _finalize_report(self, state: VPMState, metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        # aggregate metric means
        def _mean(key: str) -> float:
            vals = [m[key] for m in metrics if key in m]
            return float(sum(vals) / len(vals)) if vals else 0.0

        report = {
            "status": "ok",
            "phi": state.phi,                    # φ metrics dict
            "utility": state.utility,            # weighted sum via VPMGoal
            "mean_attention_mass": _mean("attention_mass"),
            "mean_importance_mass": _mean("importance_mass"),
            "mean_decision_mass": _mean("decision_mass"),
            "mean_att_imp_alignment": _mean("att_imp_alignment"),
        }
        return report
