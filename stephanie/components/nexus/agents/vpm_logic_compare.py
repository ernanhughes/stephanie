from __future__ import annotations
import json, uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import imageio.v2 as iio
from PIL import Image

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_processor import ScorableProcessor
from stephanie.services.zeromodel_service import ZeroModelService

from stephanie.components.nexus.vpm.state_machine import VPMState, VPMGoal, Thought, ThoughtExecutor, compute_phi
from stephanie.components.nexus.vpm.maps import MapProvider
from stephanie.components.nexus.utils.visual_thought import VisualThoughtOp, VisualThoughtType

class VPMLogicCompareAgent(BaseAgent):
    """
    Compare two cohorts (enhanced vs random) visually with logic-bootstrapped VPMs.
    Uses async metrics→VPM path, handles 1xW VPMs, and writes A/B and side-by-side films.
    """
    name = "nexus_vpm_logic_compare"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.zm: ZeroModelService = container.get("zeromodel")
        self.scorable_processor = ScorableProcessor(
            cfg=cfg,
            memory=memory,
            container=container,
            logger=logger,
        )
        self.exec = ThoughtExecutor()
        self.map_provider = MapProvider(self.zm)
        self.img_size = int(cfg.get("img_size", 256))
        self.max_steps = int(cfg.get("max_steps", 4))
        self.topk = int(cfg.get("topk", 5))
        self.min_vis_height = int(cfg.get("min_vis_height", 32))
        self.out_root = Path(str(cfg.get("out_root","runs/vpm_compare")))

    # ---------------- helpers ----------------
    def _ensure_visual_chw(self, chw: np.ndarray) -> np.ndarray:
        x = np.asarray(chw); assert x.ndim == 3
        C,H,W = x.shape
        if x.dtype != np.uint8:
            xn = x.astype(np.float32)
            if xn.max() > 1.0 or xn.min() < 0.0:
                mn, mx = float(xn.min()), float(xn.max()); rng = (mx-mn) if mx>mn else 1.0
                xn = (xn - mn) / rng
            x = (np.clip(xn, 0.0, 1.0) * 255.0).astype(np.uint8)
        if C == 1:
            x = np.tile(x, (3, 1, 1))
        if H < self.min_vis_height:
            reps = int(np.ceil(self.min_vis_height / max(1, H)))
            x = np.tile(x, (1, reps, 1))[:, :self.min_vis_height, :]
        return x

    def _hwc(self, chw: np.ndarray) -> np.ndarray:
        return np.transpose(chw, (1,2,0))

    def _augment_maps(self, maps: Dict[str, np.ndarray], X: np.ndarray) -> Dict[str, np.ndarray]:
        H, W = X.shape[-2], X.shape[-1]
        def _as01(u8_2d):
            a = u8_2d.astype(np.float32); 
            if a.max() > 1.0: a = a / 255.0
            return np.clip(a, 0.0, 1.0)
        out = dict(maps or {})
        att = _as01(X[0])
        out.setdefault("quality", att)
        out.setdefault("novelty", np.clip(np.abs(att - float(att.mean())) * 2.0, 0.0, 1.0))
        out.setdefault("uncertainty", 1.0 - att)
        if "bridge" not in out:
            bw = max(1, W // 16); mid_l = (W - bw)//2; mid_r = mid_l + bw
            center = np.zeros((H,W), dtype=np.float32); center[:, mid_l:mid_r] = 1.0
            out["bridge"] = np.clip(att * center, 0.0, 1.0)
        return out

    # --------------- pipeline ---------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        A = [Scorable.from_dict(s) for s in (context.get("scorables_enhanced") or [])]
        B = [Scorable.from_dict(s) for s in (context.get("scorables_random") or [])]
        if not A or not B:
            context[self.output_key] = {"status":"need_two_cohorts"}
            return context

        self.zm.initialize()
        run_id = f"cmp-{uuid.uuid4().hex[:8]}"
        out_dir = self.out_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        film_A = await self._process_cohort(A, out_dir / "A")
        film_B = await self._process_cohort(B, out_dir / "B")

        # side-by-side film
        sxs = []
        for fa, fb in zip(film_A, film_B):
            Ha,Wa,_ = fa.shape; Hb,Wb,_ = fb.shape
            H = max(Ha,Hb)
            ca = np.zeros((H, Wa, 3), dtype=np.uint8); ca[:Ha,:Wa] = fa
            cb = np.zeros((H, Wb, 3), dtype=np.uint8); cb[:Hb,:Wb] = fb
            sxs.append(np.concatenate([ca, cb], axis=1))
        iio.mimsave(out_dir / "compare.gif", sxs, fps=2, loop=0)

        context[self.output_key] = {"status":"ok", "run_dir": str(out_dir)}
        return context

    async def _process_cohort(self, scorables: List[Scorable], out_dir: Path) -> List[np.ndarray]:
        out_dir.mkdir(parents=True, exist_ok=True)
        frames: List[np.ndarray] = []

        # Sort by utility proxy (φ on the raw line VPM)
        scored: List[Tuple[float, Scorable]] = []
        for s in scorables:
            row = await self.scorable_processor.process(s, context={})
            mv = row.get("metrics_values", []) or []
            mc = row.get("metrics_columns", []) or []
            chw_u8, meta = await self.zm.vpm_from_scorable(s, metrics_values=mv, metrics_columns=mc)
            chw_u8 = self._ensure_visual_chw(chw_u8)
            st = VPMState(X=chw_u8, meta={"adapter": meta}, phi=compute_phi(chw_u8, {}), goal=VPMGoal({"separability":1.0,"bridge_proxy":-0.5}))
            scored.append((st.utility, s))
        scored.sort(key=lambda t: t[0], reverse=True)

        # take top K to keep films short
        topk = min(self.topk, len(scored))
        for _, s in scored[:topk]:
            frames += await self._process_one(s, out_dir)

        return frames

    async def _process_one(self, s: Scorable, out_dir: Path) -> List[np.ndarray]:
        # Build metrics + VPM
        row = await self.scorable_processor.process(s, context={})
        mv = row.get("metrics_values", []) or []
        mc = row.get("metrics_columns", []) or []
        chw_u8, meta = await self.zm.vpm_from_scorable(s, metrics_values=mv, metrics_columns=mc)
        chw_u8 = self._ensure_visual_chw(chw_u8)

        state = VPMState(X=chw_u8, meta={"adapter": meta}, phi=compute_phi(chw_u8, {}), goal=VPMGoal({"separability":1.0,"bridge_proxy":-0.5}))
        maps = self.map_provider.build(state.X).maps
        state.meta["maps"] = self._augment_maps(maps, state.X)

        film: List[np.ndarray] = [self._hwc(state.X)]

        # Step 0: logic bootstrap into channels (same pattern as refiner)
        state, *_ = self.exec.score_thought(state, Thought("logic_interesting", [
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"NOT", "a":("map","uncertainty"), "dst":1}),
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"AND", "a":("map","quality"), "b":("channel",1), "dst":0}),
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"AND", "a":("map","novelty"), "b":("channel",1), "dst":2}),
            VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"OR",  "a":("channel",0), "b":("channel",2), "dst":0}),
        ]))
        film.append(self._hwc(state.X))

        # Steps 1..N: debias + zoom
        for _ in range(self.max_steps):
            if "risk" in state.meta["maps"]:
                state, *_ = self.exec.score_thought(state, Thought("logic_debias", [
                    VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"SUB","a":("channel",0),"b":("map","risk"),"dst":0,"blend":1.0})
                ]))
            if "bridge" in state.meta["maps"]:
                state, *_ = self.exec.score_thought(state, Thought("logic_bridge_tame", [
                    VisualThoughtOp(VisualThoughtType.LOGIC, {"op":"SUB","a":("channel",0),"b":("map","bridge"),"dst":0})
                ]))

            att = state.X[0]
            y, x = np.unravel_index(np.argmax(att), att.shape)
            state, *_ = self.exec.score_thought(state, Thought("zoom_focus", [
                VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (int(x), int(y)), "scale": 2.0})
            ]))
            film.append(self._hwc(state.X))

        # Persist short film per scorable
        stem = uuid.uuid4().hex[:8]
        iio.mimsave(out_dir / f"{stem}.gif", film, fps=2, loop=0)
        return film
