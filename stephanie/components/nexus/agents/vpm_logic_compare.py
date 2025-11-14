# stephanie/components/nexus/agents/nexus_vpm_logic_compare.py
from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.nexus.vpm.maps import MapProvider
from stephanie.components.nexus.vpm.state_machine import (VPMGoal, VPMState,
                                                          compute_phi)
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_processor import ScorableProcessor
from stephanie.services.zeromodel_service import ZeroModelService

# --------------------------- helpers (pure, local) ---------------------------

def _to_rgb(u8_chw: np.ndarray) -> np.ndarray:
    X = np.asarray(u8_chw)
    assert X.ndim == 3, f"expected [C,H,W], got {X.shape}"
    C, H, W = X.shape
    if C >= 3:
        rgb = np.transpose(X[:3], (1, 2, 0))
    else:
        ch = np.transpose(X[:1], (1, 2, 0))
        rgb = np.repeat(ch, 3, axis=2)
    if rgb.dtype != np.uint8:
        mn, mx = float(rgb.min()), float(rgb.max())
        if mx - mn <= 1e-9:
            return np.zeros_like(rgb, dtype=np.uint8)
        rgb = ((rgb - mn) / (mx - mn) * 255).astype(np.uint8)
    return rgb


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
    return np.minimum(np.asarray(a01, dtype=np.float32), np.asarray(b01, dtype=np.float32))


def _logic_or(a01: np.ndarray, b01: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(a01, dtype=np.float32), np.asarray(b01, dtype=np.float32))


def _logic_sub(a01: np.ndarray, b01: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(a01, dtype=np.float32) - np.asarray(b01, dtype=np.float32), 0.0, 1.0)


def _heat_overlay(rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    H, W, _ = rgb.shape
    h = (np.clip(heat01, 0.0, 1.0) * 255).astype(np.uint8)
    h = h[:H, :W]
    overlay = np.stack([h, np.zeros_like(h), np.zeros_like(h)], axis=-1)
    out = (rgb.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _four_panel(rgb: np.ndarray, a01: np.ndarray, b01: np.ndarray, c01: np.ndarray) -> np.ndarray:
    p1 = rgb
    p2 = _heat_overlay(rgb, a01)   # attention
    p3 = _heat_overlay(rgb, b01)   # importance
    p4 = _heat_overlay(rgb, c01)   # decision
    return np.concatenate([p1, p2, p3, p4], axis=1)


def _mass(x01: np.ndarray) -> float:
    x = np.asarray(x01, dtype=np.float32)
    if x.ndim > 2:
        x = x.mean(axis=0)
    return float(np.mean(np.clip(x, 0.0, 1.0)))


def _alignment(a01: np.ndarray, b01: np.ndarray) -> float:
    A = np.asarray(a01, dtype=np.float32).ravel()
    B = np.asarray(b01, dtype=np.float32).ravel()
    num = float((A * B).sum())
    den = float(np.linalg.norm(A) * np.linalg.norm(B)) + 1e-9
    return num / den


def _occlusion_importance(
    rgb_u8: np.ndarray,
    *, patch_h: int = 12, patch_w: int = 12, stride: int = 8,
    prior: str = "top_left", channel_agg: str = "mean"
) -> np.ndarray:
    """Gradient-free occlusion importance on RGB VPM. Returns [H,W] in [0,1]."""
    H, W, _ = rgb_u8.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if prior == "top_left":
        dist = np.sqrt(yy**2 + xx**2)
        w = 1.0 - 0.3 * dist
        w[w < 0] = 0.0
        if w.max() > 0:
            w /= w.max()
    else:
        w = np.ones((H, W), dtype=np.float32)

    v01 = rgb_u8.astype(np.float32) / 255.0
    lum = v01.max(axis=2) if channel_agg == "max" else v01.mean(axis=2)
    denom = float(w.sum()) + 1e-12
    base = float((lum * w).sum() / denom)

    imp = np.zeros((H, W), dtype=np.float32)
    baseline = np.zeros_like(v01, dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2, x2 = min(H, y + patch_h), min(W, x + patch_w)
            patched = v01.copy()
            patched[y:y2, x:x2, :] = baseline[y:y2, x:x2, :]
            pl = patched.max(axis=2) if channel_agg == "max" else patched.mean(axis=2)
            occ = float((pl * w).sum() / denom)
            drop = max(0.0, base - occ)
            imp[y:y2, x:x2] += drop

    if imp.max() > 0:
        imp /= imp.max()
    return imp.astype(np.float32)


def _zoom_chw(X: np.ndarray, center_xy: Tuple[int, int], scale: float) -> np.ndarray:
    """Zoom into CHW using PIL (bicubic)."""
    X = np.asarray(X)
    C, H, W = X.shape
    cx, cy = int(center_xy[0]), int(center_xy[1])
    scale = max(1.0, float(scale))
    crop_w = max(2, int(round(W / scale)))
    crop_h = max(2, int(round(H / scale)))
    x1 = int(np.clip(cx - crop_w // 2, 0, W - crop_w))
    y1 = int(np.clip(cy - crop_h // 2, 0, H - crop_h))
    x2, y2 = x1 + crop_w, y1 + crop_h
    out = np.zeros_like(X)
    for c in range(C):
        pil = Image.fromarray((X[c]).astype(np.uint8), mode="L")
        crop = pil.crop((x1, y1, x2, y2))
        resized = crop.resize((W, H), resample=Image.BICUBIC)
        out[c] = np.asarray(resized, dtype=np.uint8)
    return out


# --------------------------------- Agent -------------------------------------

class VPMLogicCompareAgent(BaseAgent):
    name = "nexus_vpm_logic_compare"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.zm: ZeroModelService = container.get("zeromodel")
        self.map_provider = MapProvider(self.zm)

        self.img_size = int(cfg.get("img_size", 256))
        self.max_steps = int(cfg.get("max_steps", 2))     # zoom refinements per scorable
        self.out_root = Path(str(cfg.get("out_root", "runs/vpm_compare")))
        self.topk = int(cfg.get("topk", 5))
        # occlusion
        self.occ_patch_h = int(cfg.get("occ_patch_h", 12))
        self.occ_patch_w = int(cfg.get("occ_patch_w", 12))
        self.occ_stride  = int(cfg.get("occ_stride", 8))
        self.occ_prior   = str(cfg.get("occ_prior", "top_left"))
        self.occ_channel_agg = str(cfg.get("occ_channel_agg", "mean"))

        self.scorable_processor = ScorableProcessor(
            cfg=cfg,
            memory=memory,
            container=container,
            logger=logger,
        )



    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        A = [Scorable.from_dict(s) for s in (context.get("scorables_enhanced") or [])]
        B = [Scorable.from_dict(s) for s in (context.get("scorables_random") or [])]
        if not A or not B:
            context[self.output_key] = {"status": "need_two_cohorts"}
            return context

        self.zm.initialize()
        run_id = f"cmp-{uuid.uuid4().hex[:8]}"
        out_dir = self.out_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "A").mkdir(parents=True, exist_ok=True)
        (out_dir / "B").mkdir(parents=True, exist_ok=True)

        # Rank by φ-utility (goal-weighted)
        A_ranked = self._rank_by_phi(A)
        B_ranked = self._rank_by_phi(B)

        film_A, metrics_A = self._process_cohort(A_ranked[: self.topk], out_dir / "A")
        film_B, metrics_B = self._process_cohort(B_ranked[: self.topk], out_dir / "B")

        # Pad to equal length
        L = max(len(film_A), len(film_B))
        if len(film_A) < L and len(film_A) > 0:
            film_A += [film_A[-1]] * (L - len(film_A))
        if len(film_B) < L and len(film_B) > 0:
            film_B += [film_B[-1]] * (L - len(film_B))

        # Side-by-side film
        sxs = []
        for fa, fb in zip(film_A, film_B):
            Ha, Wa, _ = fa.shape
            Hb, Wb, _ = fb.shape
            H = max(Ha, Hb)
            ca = np.zeros((H, Wa, 3), dtype=np.uint8); ca[:Ha, :Wa] = fa
            cb = np.zeros((H, Wb, 3), dtype=np.uint8); cb[:Hb, :Wb] = fb
            sxs.append(np.concatenate([ca, cb], axis=1))
        iio.mimsave(out_dir / "compare.gif", sxs, fps=2, loop=0)

        # Reports
        report = self._cohort_report(metrics_A, metrics_B)
        (out_dir / "compare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._write_csv(out_dir / "compare_metrics.csv", metrics_A, metrics_B)

        context[self.output_key] = {"status": "ok", "run_dir": str(out_dir), "report": report}
        return context

    # ------------------------------- cohort ops ------------------------------

    def _rank_by_phi(self, scorables: List[Scorable]) -> List[Tuple[float, Scorable]]:
        ranked: List[Tuple[float, Scorable]] = []
        for s in scorables:
            vpm, meta = self.zm.vpm_from_scorable(s, img_size=self.img_size)
            st = VPMState(
                X=vpm,
                meta={"adapter": meta},
                phi=compute_phi(vpm, {}),
                goal=VPMGoal({"separability": 1.0, "bridge_proxy": -0.5}),
            )
            ranked.append((st.utility, s))
        ranked.sort(key=lambda t: t[0], reverse=True)
        return [s for _, s in ranked]

    def _process_cohort(self, scorables: List[Scorable], out_dir: Path) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
        frames: List[np.ndarray] = []
        metrics: List[Dict[str, float]] = []
        for s in scorables:
            f, m = self._process_one(s, out_dir)
            frames += f
            metrics.append(m)
        return frames, metrics

    def _build_maps_safe(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """MapProvider → quality/novelty/uncertainty with fallbacks."""
        H, W = X.shape[1], X.shape[2]
        try:
            built = self.map_provider.build(X)
            maps = built.maps if hasattr(built, "maps") else dict(built)
        except Exception:
            maps = {}

        if "quality" not in maps:
            q = np.clip(X.astype(np.float32).mean(axis=0) / 255.0, 0.0, 1.0)
            maps["quality"] = q
        if "novelty" not in maps:
            m = X.astype(np.float32).mean(axis=0) / 255.0
            k = max(1, min(H, W) // 64)
            if k > 1:
                # simple separable blur
                blur_h = np.apply_along_axis(lambda r: np.convolve(r, np.ones(k)/k, mode="same"), 1, m)
                blur = np.apply_along_axis(lambda c: np.convolve(c, np.ones(k)/k, mode="same"), 0, blur_h)
                nov = _normalize01(m - blur)
            else:
                nov = _normalize01(m)
            maps["novelty"] = nov
        if "uncertainty" not in maps:
            maps["uncertainty"] = _logic_not(maps["quality"])
        return {k: np.asarray(v, dtype=np.float32) for k, v in maps.items()}

    def _compose_attention(self, maps: Dict[str, np.ndarray]) -> np.ndarray:
        q = _normalize01(maps.get("quality", 0.0))
        n = _normalize01(maps.get("novelty", 0.0))
        u = _normalize01(maps.get("uncertainty", 1.0))
        not_u = _logic_not(u)
        return _logic_or(_logic_and(q, not_u), _logic_and(n, not_u))

    def _process_one(self, s: Scorable, out_dir: Path) -> Tuple[List[np.ndarray], Dict[str, float]]:
        vpm, meta = self.zm.vpm_from_scorable(s, img_size=self.img_size)
        state = VPMState(
            X=vpm,
            meta={"adapter": meta},
            phi=compute_phi(vpm, {}),
            goal=VPMGoal({"separability": 1.0, "bridge_proxy": -0.5}),
        )

        maps = self._build_maps_safe(state.X)
        rgb = _to_rgb(state.X)
        imp = _occlusion_importance(
            rgb,
            patch_h=self.occ_patch_h, patch_w=self.occ_patch_w, stride=self.occ_stride,
            prior=self.occ_prior, channel_agg=self.occ_channel_agg,
        )
        att = self._compose_attention(maps)
        # optional debias: subtract “bridge”/“risk” maps if provided
        if "bridge" in maps:
            att = _logic_sub(att, maps["bridge"])
        if "risk" in maps:
            att = _logic_sub(att, maps["risk"])

        dec = _logic_and(att, imp)

        film: List[np.ndarray] = []
        film.append(_four_panel(rgb, att, imp, dec))

        # zoom refinements
        Xz = state.X.copy()
        for _ in range(self.max_steps):
            y, x = np.unravel_index(np.argmax((att * imp)), att.shape)
            Xz = _zoom_chw(Xz, (int(x), int(y)), scale=2.0)
            # recompute maps on zoomed view
            maps_z = self._build_maps_safe(Xz)
            rgb_z = _to_rgb(Xz)
            imp_z = _occlusion_importance(
                rgb_z,
                patch_h=self.occ_patch_h, patch_w=self.occ_patch_w, stride=self.occ_stride,
                prior=self.occ_prior, channel_agg=self.occ_channel_agg,
            )
            att_z = self._compose_attention(maps_z)
            if "bridge" in maps_z:
                att_z = _logic_sub(att_z, maps_z["bridge"])
            if "risk" in maps_z:
                att_z = _logic_sub(att_z, maps_z["risk"])
            dec_z = _logic_and(att_z, imp_z)
            film.append(_four_panel(rgb_z, att_z, imp_z, dec_z))
            # roll forward for next step
            att, imp, dec = att_z, imp_z, dec_z

        # write short GIF for this scorable
        stem = uuid.uuid4().hex[:8]
        iio.mimsave(out_dir / f"{stem}.gif", film, fps=2, loop=0)

        # metrics for reporting
        metrics = {
            "decision_mass": _mass(dec),
            "attention_mass": _mass(att),
            "importance_mass": _mass(imp),
            "att_imp_alignment": _alignment(att, imp),
        }
        return film, metrics

    # --------------------------- reporting helpers ---------------------------

    def _cohort_report(self, A: List[Dict[str, float]], B: List[Dict[str, float]]) -> Dict[str, Any]:
        def mean_of(key: str, arr: List[Dict[str, float]]) -> float:
            vals = [d[key] for d in arr if key in d]
            return float(sum(vals) / len(vals)) if vals else 0.0

        keys = ["decision_mass", "attention_mass", "importance_mass", "att_imp_alignment"]
        report = {
            "status": "ok",
            "A_size": len(A),
            "B_size": len(B),
            "A_means": {k: mean_of(k, A) for k in keys},
            "B_means": {k: mean_of(k, B) for k in keys},
        }
        report["delta"] = {k: report["A_means"][k] - report["B_means"][k] for k in keys}
        # A quick scalar “separation index” you can tune later
        report["separation_index"] = float(
            0.6 * report["delta"]["decision_mass"] +
            0.3 * report["delta"]["att_imp_alignment"] +
            0.1 * report["delta"]["attention_mass"]
        )
        return report

    def _write_csv(self, path: Path, A: List[Dict[str, float]], B: List[Dict[str, float]]) -> None:
        keys = ["cohort", "decision_mass", "attention_mass", "importance_mass", "att_imp_alignment"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for d in A:
                w.writerow({"cohort": "A", **{k: d.get(k, 0.0) for k in keys if k != "cohort"}})
            for d in B:
                w.writerow({"cohort": "B", **{k: d.get(k, 0.0) for k in keys if k != "cohort"}})
