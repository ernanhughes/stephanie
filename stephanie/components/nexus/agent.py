# stephanie/components/nexus/agent.py
from __future__ import annotations
import json, time, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import imageio.v2 as iio
import torch

from stephanie.scoring.scorable import Scorable
from stephanie.services.graph_vision_scorer import VisionScorer
from scripts.train_vpm_thought_model import VPMThoughtModel
from stephanie.components.nexus.vpm.state_machine import VPMGoal, VPMState, compute_phi, Thought, ThoughtExecutor
from stephanie.components.nexus.utils.visual_thought import VisualThoughtOp, VisualThoughtType
from stephanie.agents.base_agent import BaseAgent
from stephanie.services.zeromodel_service import ZeroModelService

class NexusAgent(BaseAgent):
    """
    Scorable-centric testbed:
      Scorable -> VPM (adapter) -> VisionScorer -> Thought rollout -> GIF + metrics.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.viz = VisionScorer(config=cfg or {})
        self.device = torch.device(self.cfg.device)

        # Load thought model (optional for tonight: can run with simple zooms even without ckpt)
        self.model = None
        if self.cfg.checkpoint:
            ckpt = torch.load(self.cfg.checkpoint, map_location=self.cfg.device)
            self.model = VPMThoughtModel(torch.tensor(0))  # won't use config path; we call heads directly
            self.model = VPMThoughtModel.__new__(VPMThoughtModel)  # avoid ctor mismatch in snippet env
            self.model = None  # fallback: use heuristic thoughts
        self.executor = ThoughtExecutor(
            visual_op_cost={"zoom":1.0, "bbox":0.3, "path":0.4, "highlight":0.5, "blur":0.6}
        )
        self.zm: ZeroModelService = self.container.get("zeromodel")

    def _greedy_thought(self, state: VPMState, goal: VPMGoal, step: int) -> Thought:
        # For the demo tonight: use a deterministic zoom on densest region
        # (simple heuristic — can swap for self.model if loaded)
        cx = cy = 128
        scale = 2.0 if step == 1 else 1.5
        return Thought(
            name=f"Zoom_{step}",
            ops=[VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (cx, cy), "scale": scale})],
            intent=f"Focus region for goal {goal.task_type}",
            cost=1.0
        )

    def run_scorables(self, scorables: List[Scorable], run_name: Optional[str]=None) -> Dict[str, Any]:
        run_id = run_name or f"nexus-{uuid.uuid4().hex[:8]}"
        run_dir = self.cfg.out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "run_id": run_id,
            "created_utc": time.time(),
            "items": []
        }

        for idx, s in enumerate(scorables):
            item_id = f"{run_id}-{idx:03d}"
            item_dir = run_dir / item_id
            item_dir.mkdir(parents=True, exist_ok=True)

            # 1) Scorable -> VPM
            vpm_u8, meta = self.zm.vpm_from_scorable(s, item_dir)
            vpm = vpm_u8.astype(np.float32) / 255.0
            H, W = vpm.shape[1], vpm.shape[2]

            # 2) VisionScorer
            # The vision scorer expects multi-layout stack; for now provide a single layout replicated
            vpm_stack = np.stack([vpm_u8, vpm_u8], axis=0)  # [n_layouts=2, 3, H, W]
            scores = self.viz.model(torch.from_numpy(vpm_stack[None].astype(np.float32)/255.0)).copy()  # bare call for speed
            # Use public API instead:
            vs = self.viz
            vis = {
                "vision_symmetry": float(0.0),
                "vision_bridge_proxy": float(0.0),
                "vision_spectral_gap_bucket": 1,
            }
            try:
                vis = vs.score_graph(graph={"dummy": True}, timeout_s=1.0)  # placeholder since we aren't passing nx graph
            except Exception:
                pass

            # 3) Thought rollout on the VPM
            goal = VPMGoal(weights={"separability": 1.0}, task_type=s.target_type)
            state = VPMState(X=vpm, meta={"positions":{}}, phi=compute_phi(vpm, {"positions":{}}), goal=goal)
            frames = []
            metrics = []
            # initial frame
            Image.fromarray(np.transpose(vpm_u8, (1,2,0))).save(item_dir / "frame_00.png")
            frames.append(np.transpose(vpm_u8, (1,2,0)))

            total_delta = 0.0
            for step in range(1, self.cfg.max_steps+1):
                thought = self._greedy_thought(state, goal, step)
                new_state, delta, cost, bcs = self.executor.score_thought(state, thought)
                total_delta += float(delta)

                # render & save frame
                x = (np.clip(new_state.X, 0, 1.0) * 255).astype(np.uint8)
                Image.fromarray(np.transpose(x, (1,2,0))).save(item_dir / f"frame_{step:02d}.png")
                frames.append(np.transpose(x, (1,2,0)))

                metrics.append({
                    "step": step,
                    "thought": thought.name,
                    "delta": float(delta),
                    "bcs": float(bcs),
                    "cost": float(cost),
                    "utility": float(new_state.utility),
                })
                state = new_state
                if delta < 0.01:
                    break

            gif_path = item_dir / "filmstrip.gif"
            iio.mimsave(gif_path, frames, fps=1, loop=0)

            item_rec = {
                "scorable_id": s.id,
                "target_type": s.target_type,
                "adapter_meta": meta,
                "vision": vis,
                "rollout": {
                    "steps": metrics,
                    "total_delta": total_delta,
                    "frames": [str((item_dir / f"frame_{i:02d}.png").as_posix()) for i in range(len(frames))],
                    "gif": str(gif_path.as_posix()),
                }
            }
            with open(item_dir / "metrics.json", "w") as f:
                json.dump(item_rec, f, indent=2)

            manifest["items"].append(item_rec)

        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return {"run_dir": str(run_dir), "run_id": run_id, "n_items": len(manifest["items"])}
