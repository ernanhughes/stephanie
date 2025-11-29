# stephanie/components/nexus/agents/vpm_logic_compare.py
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.nexus.utils.visual_thought import (VisualThoughtOp,
                                                             VisualThoughtType)
from stephanie.components.nexus.vpm.maps import MapProvider
from stephanie.components.nexus.vpm.state_machine import (Thought,
                                                          ThoughtExecutor,
                                                          VPMGoal, VPMState,
                                                          compute_phi)
from stephanie.scoring.metrics.scorable_processor import ScorableProcessor
from stephanie.scoring.scorable import Scorable
from stephanie.services.zeromodel_service import ZeroModelService

log = logging.getLogger(__name__)


class VPMLogicCompareAgent(BaseAgent):
    """
    Compare two cohorts (enhanced vs random) using logic-bootstrapped VPMs.
    - Async metrics→VPM path
    - Handles 1xW line VPMs (tiles to min height; 3ch ensure)
    - Refiner-style logs (φ before/after, Δutility, cost, BCS) with auto REVERT
    - Per-scorable films (top: original, bottom: refined + % utility bar)
    - Side-by-side A/B film (length-equalized)
    - Cohort summary JSON
    """
    name = "nexus_vpm_logic_compare"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.zm: ZeroModelService = container.get("zeromodel")
        self.scorable_processor = ScorableProcessor(cfg=cfg, memory=memory, container=container, logger=logger)
        self.exec = ThoughtExecutor(visual_op_cost={"zoom":1.0, "logic":0.2, "bbox":0.3, "path":0.4, "highlight":0.5, "blur":0.6})
        self.map_provider = MapProvider(self.zm)

        self.img_size = int(cfg.get("img_size", 256))
        self.max_steps = int(cfg.get("max_steps", 4))
        self.topk = int(cfg.get("topk", 5))
        self.min_vis_height = int(cfg.get("min_vis_height", 32))
        self.utility_threshold = float(cfg.get("utility_threshold", 0.98))  # early stop
        self.out_root = Path(str(cfg.get("out_root", "runs/vpm_compare")))

    # ------------------------------ public ---------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        A = [Scorable.from_dict(s) for s in (context.get("scorables_targeted") or [])]
        B = [Scorable.from_dict(s) for s in (context.get("scorables_baseline") or [])]
        if not A or not B:
            context[self.output_key] = {"status": "need_two_cohorts"}
            log.warning("Need two cohorts for comparison: A=%d, B=%d", len(A), len(B))
            return context

        self.zm.initialize()
        run_id = context.get("pipeline_run_id")
        out_dir = self.out_root / run_id
        (out_dir / "A").mkdir(parents=True, exist_ok=True)
        (out_dir / "B").mkdir(parents=True, exist_ok=True)
        log.info("=== VPMLogicCompare run_id=%s out_dir=%s ===", run_id, out_dir)

        # Build both cohorts
        film_A, summary_A = await self._process_cohort(A, out_dir / "A", label="A")
        film_B, summary_B = await self._process_cohort(B, out_dir / "B", label="B")

        # Equalize film length for side-by-side
        L = max(len(film_A), len(film_B))
        if film_A:
            film_A += [film_A[-1]] * (L - len(film_A))
        if film_B:
            film_B += [film_B[-1]] * (L - len(film_B))

        # Side-by-side movie
        sxs = []
        for fa, fb in zip(film_A, film_B):
            Ha, Wa, _ = fa.shape
            Hb, Wb, _ = fb.shape
            H = max(Ha, Hb)
            ca = np.zeros((H, Wa, 3), dtype=np.uint8); ca[:Ha, :Wa] = fa
            cb = np.zeros((H, Wb, 3), dtype=np.uint8); cb[:Hb, :Wb] = fb
            frame = np.concatenate([ca, cb], axis=1)
            sxs.append(frame)

        iio.mimsave(out_dir / "compare.gif", sxs, fps=2, loop=0)

        # Persist summary
        summary = {"A": summary_A, "B": summary_B}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        context[self.output_key] = {"status": "ok", "run_dir": str(out_dir), "frames": len(sxs)}
        log.info("Saved compare.gif (%d frames) → %s", len(sxs), out_dir / "compare.gif")
        return context

    # ------------------------------ cohort ----------------------------------

    async def _process_cohort(self, scorables: List[Scorable], out_dir: Path, label: str) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Returns (all_frames_for_cohort, per_item_summary_list).
        We sort items by initial utility (descending) and take top-k.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        all_frames: List[np.ndarray] = []
        item_summaries: List[Dict[str, Any]] = []

        # Score raw line VPMs for ranking
        scored: List[Tuple[float, Scorable, Dict[str, Any]]] = []
        for s in scorables:
            row = await self.scorable_processor.process(s, context={})
            mv = row.get("metrics_values", []) or []
            mc = row.get("metrics_columns", []) or []
            chw_u8, meta = await self.zm.vpm_from_scorable(s, metrics_values=mv, metrics_columns=mc)
            chw_u8 = self._ensure_visual_chw(chw_u8)
            st = VPMState(X=chw_u8, meta={"adapter": meta}, phi=compute_phi(chw_u8, {}), goal=self._default_goal())
            scored.append((float(st.utility), s, {"phi": st.phi, "shape": tuple(chw_u8.shape)}))

        scored.sort(key=lambda t: t[0], reverse=True)
        pick = scored[: min(self.topk, len(scored))]

        for rank, (_u, s, meta0) in enumerate(pick, start=1):
            frames, item_meta = await self._process_one(s, out_dir)
            all_frames.extend(frames)
            item_summaries.append({"rank": rank, "id": getattr(s, "id", None), "initial": meta0, **item_meta})

        # write cohort index
        (out_dir / "index.json").write_text(json.dumps(item_summaries, indent=2), encoding="utf-8")
        log.info("[%s] wrote %s", label, out_dir / "index.json")
        return all_frames, item_summaries

    # ------------------------------ single ----------------------------------

    async def _process_one(self, s: Scorable, out_dir: Path) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        # Build metrics + VPM
        row = await self.scorable_processor.process(s, context={})
        mv = row.get("metrics_values", []) or []
        mc = row.get("metrics_columns", []) or []
        chw_u8, meta = await self.zm.vpm_from_scorable(s, metrics_values=mv, metrics_columns=mc)
        log.info("[item %s] adapter VPM: shape=%s", getattr(s, "id", None), getattr(chw_u8, "shape", None))
        chw_u8 = self._ensure_visual_chw(chw_u8)

        # State + φ
        state = VPMState(X=chw_u8, meta={"adapter": meta}, phi=compute_phi(chw_u8, {}), goal=self._default_goal())

        # Maps (safe fallback)
        try:
            maps = self.map_provider.build(state.X).maps
            log.info("[item %s] MapProvider maps: %s", getattr(s, "id", None), sorted(list(maps.keys())))
        except Exception as e:
            log.warning("[item %s] MapProvider.build failed: %s; using fallbacks.", getattr(s, "id", None), e)
            maps = {}
        state.meta["maps"] = self._augment_maps(maps, state.X)
        self._alias_risk_map(state)  # ensure 'risk' exists when only 'uncertainty' exists

        # Film assembly (refiner style)
        initial_rgb = self._hwc(chw_u8)
        frames: List[np.ndarray] = [self._compose_frame(initial_rgb, self._hwc(state.X), state.utility)]

        # Step 0: logic bootstrap (same as refiner)
        state, rec = self._apply_and_log(state, "bootstrap", [
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"NOT", "a":("map","uncertainty"), "dst":1}),
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"AND", "a":("map","quality"), "b":("channel",1), "dst":0}),
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"AND", "a":("map","novelty"), "b":("channel",1), "dst":2}),
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"OR",  "a":("channel",0), "b":("channel",2), "dst":0}),
        ])
        frames.append(self._compose_frame(initial_rgb, self._hwc(state.X), state.utility))
        records = [rec]

        # Steps 1..N: debias/bridge + zoom with acceptance gating and early-stop
        for step in range(self.max_steps):
            if state.utility >= self.utility_threshold:
                log.info("[item %s] early-stop: step=%d utility=%.4f ≥ %.4f",
                         getattr(s, "id", None), step, state.utility, self.utility_threshold)
                break

            # debias (uses 'risk' map; alias provides from uncertainty if missing)
            state, rec = self._apply_and_log(state, "debias", [
                VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"SUB","a":("channel",0),"b":("map","risk"),"dst":0,"blend":1.0})
            ], skip_if_missing=("map","risk"))
            if rec: records.append(rec); frames.append(self._compose_frame(initial_rgb, self._hwc(state.X), state.utility))

            # bridge-tame
            state, rec = self._apply_and_log(state, "bridge_tame", [
                VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"SUB","a":("channel",0),"b":("map","bridge"),"dst":0})
            ], skip_if_missing=("map","bridge"))
            if rec: records.append(rec); frames.append(self._compose_frame(initial_rgb, self._hwc(state.X), state.utility))

            # zoom-to-attention (with revert if BCS < 0)
            att = state.X[0]
            y, x = np.unravel_index(np.argmax(att), att.shape)
            log.info("[item %s] zoom_focus target: center(x=%d,y=%d) scale=2.0", getattr(s, "id", None), int(x), int(y))
            state, rec = self._apply_and_log(state, "zoom_focus", [
                VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (int(x), int(y)), "scale": 2.0})
            ])
            records.append(rec); frames.append(self._compose_frame(initial_rgb, self._hwc(state.X), state.utility))

        # Persist per-item film and summary
        stem = uuid.uuid4().hex[:8]
        gif_path = out_dir / f"{stem}.gif"
        iio.mimsave(gif_path, frames, fps=2, loop=0)

        item_meta = {
            "gif": str(gif_path),
            "initial_phi": records[0]["phi_before"],
            "final_phi": records[-1]["phi_after"],
            "initial_utility": records[0]["utility_before"],
            "final_utility": records[-1]["utility_after"],
            "records": records,
        }
        return frames, item_meta

    # ------------------------------ helpers ---------------------------------

    def _default_goal(self) -> VPMGoal:
        return VPMGoal(weights={"separability": 1.0, "bridge_proxy": -0.5})

    def _alias_risk_map(self, state: VPMState) -> None:
        m = state.meta.get("maps", {})
        if "risk" not in m and "uncertainty" in m:
            m["risk"] = m["uncertainty"]
            state.meta["maps"] = m
            log.info("[maps] aliasing 'risk' → 'uncertainty' (fallback)")

    def _log_state(self, tag: str, phi: Dict[str, float], util: float) -> None:
        log.info(
            "[%s] φ: sep=%.4f bridge=%.4f spec_gap=%.4f symmetry=%.4f crossings=%s | utility=%.4f",
            tag,
            float(phi.get("separability", 0.0)),
            float(phi.get("bridge_proxy", 0.0)),
            float(phi.get("spectral_gap", 0.0)),
            float(phi.get("vision_symmetry", 0.0)),
            str(phi.get("crossings", 0)),
            float(util),
        )

    def _apply_and_log(
        self,
        state: VPMState,
        name: str,
        ops: List[VisualThoughtOp],
        skip_if_missing: Tuple[str, str] | None = None,
    ):
        # Optional skip if a required ("map", key) is missing
        if skip_if_missing:
            src_kind, key = skip_if_missing
            if src_kind == "map" and key not in state.meta.get("maps", {}):
                log.info("[%s] skip op due to missing map: %s", name, key)
                return state, None

        before_phi = dict(state.phi)
        u0 = float(state.utility)
        self._log_state(f"{name}:before", before_phi, u0)

        new_state, delta, cost, bcs = self.exec.score_thought(state, Thought(name, ops))
        u1 = float(new_state.utility)
        after_phi = dict(new_state.phi)

        # Accept/reject with revert by BCS (benefit minus cost)
        if (u1 - u0) < 0.0 and bcs < 0.0:
            log.info("[%s:rejected] Δutility=%+.4f cost=%.3f bcs=%+.4f → REVERT", name, (u1 - u0), cost, bcs)
            self._log_state(f"{name}:after(REVERTED)", before_phi, u0)
            return state, {
                "name": name,
                "phi_before": before_phi,
                "phi_after": before_phi,
                "utility_before": u0,
                "utility_after": u0,
                "delta_utility": 0.0,
                "cost": float(cost),
                "bcs": float(bcs),
                "reverted": True,
            }

        log.info(
            "[%s:after] Δφ: sep=%+0.4f bridge=%+0.4f spec_gap=%+0.4f symmetry=%+0.4f | Δutility=%+0.4f cost=%.3f bcs=%+.4f | utility=%.4f",
            name,
            float(after_phi.get("separability", 0.0)) - float(before_phi.get("separability", 0.0)),
            float(after_phi.get("bridge_proxy", 0.0)) - float(before_phi.get("bridge_proxy", 0.0)),
            float(after_phi.get("spectral_gap", 0.0)) - float(before_phi.get("spectral_gap", 0.0)),
            float(after_phi.get("vision_symmetry", 0.0)) - float(before_phi.get("vision_symmetry", 0.0)),
            (u1 - u0),
            float(cost),
            float(bcs),
            u1,
        )
        return new_state, {
            "name": name,
            "phi_before": before_phi,
            "phi_after": after_phi,
            "utility_before": u0,
            "utility_after": u1,
            "delta_utility": (u1 - u0),
            "cost": float(cost),
            "bcs": float(bcs),
            "reverted": False,
        }

    def _ensure_visual_chw(self, chw: np.ndarray) -> np.ndarray:
        x = np.asarray(chw); assert x.ndim == 3
        C, H, W = x.shape
        if x.dtype != np.uint8:
            xn = x.astype(np.float32)
            mn, mx = float(xn.min()), float(xn.max())
            rng = (mx - mn) if mx > mn else 1.0
            xn = (xn - mn) / rng
            x = (np.clip(xn, 0.0, 1.0) * 255.0).astype(np.uint8)
        if C == 1:
            x = np.tile(x, (3, 1, 1))
        if H < self.min_vis_height:
            reps = int(np.ceil(self.min_vis_height / max(1, H)))
            x = np.tile(x, (1, reps, 1))[:, : self.min_vis_height, :]
        return x

    def _augment_maps(self, maps: Dict[str, np.ndarray], X: np.ndarray) -> Dict[str, np.ndarray]:
        H, W = X.shape[-2], X.shape[-1]

        def _as01(u8_2d):
            a = u8_2d.astype(np.float32)
            if a.max() > 1.0:
                a = a / 255.0
            return np.clip(a, 0.0, 1.0)

        out = dict(maps or {})
        att = _as01(X[0])  # attention proxy
        out.setdefault("quality", att)
        out.setdefault("novelty", np.clip(np.abs(att - float(att.mean())) * 2.0, 0.0, 1.0))
        out.setdefault("uncertainty", 1.0 - att)

        if "bridge" not in out:
            bw = max(1, W // 16)
            center = np.zeros((H, W), dtype=np.float32)
            mid_l = (W - bw) // 2
            mid_r = mid_l + bw
            center[:, mid_l:mid_r] = 1.0
            out["bridge"] = np.clip(att * center, 0.0, 1.0)

        return out

    def _hwc(self, chw: np.ndarray) -> np.ndarray:
        return np.transpose(chw, (1, 2, 0))

    def _compose_frame(self, top_rgb: np.ndarray, bottom_rgb: np.ndarray, utility: float) -> np.ndarray:
        """
        Compose a blog-friendly frame:
          [ top  : original ]
          [ bottom: current refined + "Utility: XX%" bar ]
        """
        assert top_rgb.shape == bottom_rgb.shape, "top/bottom dims must match"
        H, W, _ = top_rgb.shape
        composite = np.vstack([top_rgb, bottom_rgb]).astype(np.uint8)
        bar_h = max(18, int(0.12 * H))

        canvas = np.zeros((2 * H + bar_h, W, 3), dtype=np.uint8)
        canvas[: 2 * H] = composite

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # bar + label
        draw.rectangle([(0, 2 * H), (W, 2 * H + bar_h)], fill=(0, 0, 0))
        pct = int(np.clip(utility, 0.0, 1.0) * 100.0 + 0.5)
        label = f"Utility: {pct}%"
        try:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        except Exception:
            tw, th = draw.textsize(label, font=font)
        x, y = 10, 2 * H + (bar_h - th) // 2
        draw.text((x + 1, y + 1), label, fill=(0, 0, 0), font=font)
        draw.text((x, y), label, fill=(255, 255, 255), font=font)

        return np.asarray(img)
