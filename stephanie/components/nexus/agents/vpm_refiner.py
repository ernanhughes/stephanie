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
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_processor import ScorableProcessor
from stephanie.services.zeromodel_service import ZeroModelService

from stephanie.components.nexus.utils.visual_thought import VisualThoughtOp, VisualThoughtType
from stephanie.components.nexus.vpm.state_machine import (
    VPMState, VPMGoal, Thought, ThoughtExecutor, compute_phi
)
from stephanie.components.nexus.vpm.maps import MapProvider
from stephanie.utils.vpm_utils import ensure_chw_u8, detect_vpm_layout, vpm_quick_dump

log = logging.getLogger(__name__)


@dataclass
class VPMRefinerConfig:
    mode: str = "filmstrip"         # "online" | "filmstrip"
    out_root: str = "runs/vpm"
    img_size: int = 256             # target visual size; source can be 1xW
    max_steps: int = 10
    utility_threshold: float = 0.90 # stop when state.utility >= this
    min_vis_height: int = 32        # expand 1xW → HxW for visibility

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

# ----------------------------- The Agent -------------------------------------

class VPMRefinerAgent(BaseAgent):
    """
    VPM Thought Refiner (metrics → VPM → logic bootstrap → zoom/refine).
    Works with the new async ZeroModelService.vpm_from_scorable API that requires
    metrics from ScorableProcessor. Well if you had my goods the money you need to **** make
    """
    name = "nexus_vpm_refiner"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = VPMRefinerConfig(
            mode=str(cfg.get("mode", "filmstrip")),
            out_root=str(cfg.get("out_root", "runs/vpm_refiner")),
            img_size=int(cfg.get("img_size", 256)),
            max_steps=int(cfg.get("max_steps", 10)),
            utility_threshold=float(cfg.get("utility_threshold", 0.90)),
            min_vis_height=int(cfg.get("min_vis_height", 32)),
            occ_patch_h=int(cfg.get("occ_patch_h", 12)),
            occ_patch_w=int(cfg.get("occ_patch_w", 12)),
            occ_stride=int(cfg.get("occ_stride", 8)),
            occ_prior=str(cfg.get("occ_prior", "top_left")),
            occ_channel_agg=str(cfg.get("occ_channel_agg", "mean")),
        )
        # Order matters: get services first
        self.zm: ZeroModelService = container.get("zeromodel")
        self.scorable_processor = ScorableProcessor(
            cfg=cfg,
            memory=memory,
            container=container,
            logger=logger,
        )

        self.exec = ThoughtExecutor(
            visual_op_cost={"zoom":1.0, "bbox":0.3, "path":0.4, "highlight":0.5, "blur":0.6, "logic":0.2}
        )
        self.map_provider = MapProvider(self.zm)

    # ------------------------------ run ------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        scorables = list(context.get("scorables") or [])
        if not scorables:
            context[self.output_key] = {"status": "no_scorables"}
            return context
        self.zm.initialize()

        seed = Scorable.from_dict(scorables[0])

        run_id = context.get("pipeline_run_id")
        run_dir = Path(self.cfg.out_root) / f"{run_id}"

        # 1) Compute metrics for this scorable
        row = await self.scorable_processor.process(seed, context=context)

        metrics_values = row.get("metrics_values", []) or []
        metrics_columns = row.get("metrics_columns", []) or []

        # 2) Prepare ZeroModel + build VPM (async API)
        chw_u8, adapter_meta = await self.zm.vpm_from_scorable(
            seed, metrics_values=metrics_values, metrics_columns=metrics_columns
        )  # CHW uint8, typically 1xW per doc
        log.info("VPM layout after vpm_from_scorable: %s", detect_vpm_layout(chw_u8))
        
        info = vpm_quick_dump(chw_u8, run_dir / "vpm_debug", "sample")
        log.info("VPM quick dump: shape=%s gray_path=%s", info["shape"], info["gray_path"])

        # 3) Make visual (expand/tile, ensure 3ch)
        chw_u8 = self._ensure_visual_chw(chw_u8)
        log.info("VPM layout after visual ensure: %s", detect_vpm_layout(chw_u8))

        # 4) State + maps
        state = VPMState(
            X=chw_u8,
            meta={"adapter": adapter_meta},
            phi=compute_phi(chw_u8, {}),
            goal=VPMGoal(weights={"separability": 1.0, "bridge_proxy": -0.5}),
        )

        
        # MapProvider can compute derived maps (works on CHW uint8)
        try:
            maps = self.map_provider.build(state.X).maps
        except Exception as e:
            log.warning("MapProvider.build failed (%s); using safe fallbacks.", e)
            maps = {}

        state.meta["maps"] = self._augment_maps(maps, state.X)
        frames: List[np.ndarray] = [self._hwc(state.X)]
        steps_meta: List[Dict[str, Any]] = []

        # Bootstrap with logic: interesting = (quality ∧ ¬uncert) ∨ (novelty ∧ ¬uncert)
        state, _, _, _ = self.exec.score_thought(state, Thought("logic_bootstrap", [
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op": "NOT", "a": ("map", "uncertainty"), "dst": 1}),
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op": "AND", "a": ("map", "quality"), "b": ("channel", 1), "dst": 0}),
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op": "AND", "a": ("map", "novelty"), "b": ("channel", 1), "dst": 2}),
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op": "OR",  "a": ("channel", 0), "b": ("channel", 2), "dst": 0}),
        ]))
        frames.append(self._hwc(state.X))

        # 5) Iterate: debias + zoom-to-attention
        for step in range(self.cfg.max_steps):
            if state.utility >= self.cfg.utility_threshold:
                break

            if "risk" in state.meta["maps"]:
                state, _, _, _ = self.exec.score_thought(state, Thought("logic_debias", [
                    VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"SUB","a":("channel",0),"b":("map","risk"),"dst":0,"blend":1.0})
                ]))
            if "bridge" in state.meta["maps"]:
                state, _, _, _ = self.exec.score_thought(state, Thought("logic_bridge_tame", [
                    VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"SUB","a":("channel",0),"b":("map","bridge"),"dst":0})
                ]))

            att = state.X[0]
            y, x = np.unravel_index(np.argmax(att), att.shape)
            prev_u = state.utility
            state, _, _, _ = self.exec.score_thought(state, Thought("zoom_focus", [
                VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (int(x), int(y)), "scale": 2.0})
            ]))

            steps_meta.append({"step": step, "utility_before": float(prev_u), "utility_after": float(state.utility)})
            frames.append(self._hwc(state.X))

        # 6) Persist film/metrics (optional)
        if self.cfg.mode == "filmstrip":
            run_dir.mkdir(parents=True, exist_ok=True)
            for i, fr in enumerate(frames):
                Image.fromarray(fr).save(run_dir / f"frame_{i:02d}.png")
            iio.mimsave(run_dir / "filmstrip.gif", frames, fps=2, loop=0)
            (run_dir / "metrics.json").write_text(json.dumps({"steps": steps_meta}, indent=2), encoding="utf-8")
            context["vpm_artifacts"] = {"run_dir": str(run_dir)}

        context[self.output_key] = {"status": "ok", "final_utility": float(state.utility), "steps": steps_meta}
        print("VPMRefinerAgent completed; final utility=%.4f" % state.utility)
        print("Artifacts dir:", run_dir)
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


    # ---------------------------- helpers ----------------------------

    def _ensure_visual_chw(self, chw: np.ndarray) -> np.ndarray:
        """
        Ensure the VPM is CHW uint8, >=3 channels, and visually legible when H==1.
        - If C==1, tile to 3
        - If H==1, vertically tile to min_vis_height
        """
        X = ensure_chw_u8(chw, force_three=True)
        C, H, W = X.shape
        if H == 1 and self.cfg.min_vis_height > 1:
            reps = max(1, int(np.ceil(self.cfg.min_vis_height / float(H))))
            X = np.tile(X, (1, reps, 1))[:, : self.cfg.min_vis_height, :]
        return X
    
    
    def _hwc(self, chw: np.ndarray) -> np.ndarray:
        return np.transpose(chw, (1, 2, 0))

    # Build a minimal map set if MapProvider lacks some entries
    def _augment_maps(self, maps: Dict[str, np.ndarray], X: np.ndarray) -> Dict[str, np.ndarray]:
        H, W = X.shape[-2], X.shape[-1]
        def _as01(u8_2d):
            a = u8_2d.astype(np.float32)
            if a.max() > 1.0: a = a / 255.0
            return np.clip(a, 0.0, 1.0)

        out = dict(maps or {})
        # Provide light-weight fallbacks from channels
        att = _as01(X[0])  # attention proxy
        if "quality" not in out:
            out["quality"] = att
        if "novelty" not in out:
            # crude novelty proxy: local contrast vs global mean
            out["novelty"] = np.clip(np.abs(att - float(att.mean())) * 2.0, 0.0, 1.0)
        if "uncertainty" not in out:
            out["uncertainty"] = 1.0 - att
        if "bridge" not in out:
            # center band intensity as simple "bridge" proxy
            bw = max(1, W // 16)
            center = np.zeros((H, W), dtype=np.float32)
            mid_l = (W - bw) // 2; mid_r = mid_l + bw
            center[:, mid_l:mid_r] = 1.0
            out["bridge"] = np.clip(att * center, 0.0, 1.0)
        return out


def _heat_overlay(rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a heat map (single channel [H,W] in [0,1]) onto an RGB image."""
    H, W, _ = rgb.shape
    h = (np.clip(heat01, 0.0, 1.0) * 255).astype(np.uint8)
    h = h[:H, :W]
    # red channel heat
    h_rgb = np.stack([h, np.zeros_like(h), np.zeros_like(h)], axis=-1)
    out = (rgb.astype(np.float32) * (1 - alpha) + h_rgb.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


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

