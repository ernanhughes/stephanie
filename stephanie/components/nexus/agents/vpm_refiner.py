# stephanie/components/nexus/agents/nexus_vpm_refiner.py
from __future__ import annotations
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import imageio.v2 as iio

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.services.zeromodel_service import ZeroModelService
from stephanie.components.nexus.vpm.state_machine import (
    VPMState,
    VPMGoal,
    ThoughtExecutor,
    compute_phi,
)

log = logging.getLogger(__name__)


@dataclass
class VPMRefinerConfig:
    mode: str = "online"  # "online" | "filmstrip"
    out_root: str = "runs/vpm"
    img_size: int = 256
    max_steps: int = 10
    phi_threshold: float = 0.90
    dims: List[str] = None

    def __post_init__(self):
        self.dims = self.dims or ["clarity", "coverage"]


class VPMRefinerAgent(BaseAgent):
    """
    Agent 01: VPM Thought Refiner
    - Converts a scorable to a VPM
    - Iteratively applies visual thought ops
    - Stops when phi >= threshold (or no Δphi)
    - Optionally writes a filmstrip for auditing
    """

    name = "nexus_vpm_refiner"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = VPMRefinerConfig(
            mode=str(cfg.get("mode", "online")),
            out_root=str(cfg.get("out_root", "runs/vpm")),
            img_size=int(cfg.get("img_size", 256)),
            max_steps=int(cfg.get("max_steps", 10)),
            phi_threshold=float(cfg.get("phi_threshold", 0.90)),
            dims=list(cfg.get("dims", ["clarity", "coverage"])),
        )
        self.executor = ThoughtExecutor(
            visual_op_cost={
                "zoom": 1.0,
                "bbox": 0.3,
                "path": 0.4,
                "highlight": 0.5,
                "blur": 0.6,
            }
        )
        self.zm: ZeroModelService = container.get("zeromodel")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        scorables = list(context.get("scorables") or [])
        if not scorables:
            context[self.output_key] = {"status": "no_scorables"}
            return context

        seed = Scorable.from_dict(scorables[0])
        self.zm.initialize()

        vpm_u8, adapter_meta = self.zm.vpm_from_scorable(
            seed, img_size=self.cfg.img_size
        )
        state = VPMState(
            vpm=vpm_u8,
            scorable=seed,
            history=[],
            metadata={"adapter": adapter_meta},
        )
        goal = VPMGoal(
            dimensions=self.cfg.dims, target_value=self.cfg.phi_threshold
        )

        frames = [np.transpose(vpm_u8, (1, 2, 0))]
        steps_meta: List[Dict[str, Any]] = []

        for step in range(self.cfg.max_steps):
            phi = compute_phi(state, goal)
            if phi >= 1.0:
                break

            thought = self.executor.select_thought(state, goal, step)
            if not thought:
                break

            new_vpm = state.vpm
            for op in thought.ops:
                new_vpm = self.zm.apply_visual_op(new_vpm, op)

            vision_scores = self.zm.score_vpm_image(new_vpm, goal.dimensions)
            new_state = VPMState(
                vpm=new_vpm,
                scorable=seed,  # keep same text; ops are attentional by default
                history=state.history + [thought],
                metadata={
                    **state.metadata,
                    f"t{step}": thought.to_dict(),
                    "vision_scores": vision_scores,
                },
            )
            delta_phi = compute_phi(new_state, goal) - phi

            steps_meta.append(
                {
                    "step": step,
                    "thought": thought.name,
                    "ops": [op.to_dict() for op in thought.ops],
                    "phi_before": float(phi),
                    "phi_after": float(phi + delta_phi),
                    "delta_phi": float(delta_phi),
                }
            )

            state = new_state
            if self.cfg.mode == "filmstrip":
                frames.append(np.transpose(new_vpm, (1, 2, 0)))

            if delta_phi <= 1e-3:  # no more progress
                break

        # persist filmstrip/metrics if requested
        if self.cfg.mode == "filmstrip":
            run_id = f"vpm-{uuid.uuid4().hex[:8]}"
            run_dir = Path(self.cfg.out_root) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            for i, fr in enumerate(frames):
                Image.fromarray(fr).save(run_dir / f"frame_{i:02d}.png")
            iio.mimsave(run_dir / "filmstrip.gif", frames, fps=1, loop=0)
            (run_dir / "metrics.json").write_text(
                json.dumps({"steps": steps_meta}, indent=2), encoding="utf-8"
            )
            context["vpm_artifacts"] = {"run_dir": str(run_dir)}

        context[self.output_key] = {
            "status": "ok",
            "final_phi": float(compute_phi(state, goal)),
            "steps": steps_meta,
            "history_len": len(state.history),
        }
        return context
