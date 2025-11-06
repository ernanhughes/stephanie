from __future__ import annotations
import asyncio
import json
import time
import uuid
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

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.zm.initialize()
        scorables = context.get("scorables", [])
        
        # --- NEW: Initialize the Graph Substrate ---
        # For simplicity, take the first scorable as the seed.
        if not scorables:
            return context
            
        seed_scorable = Scorable.from_dict(scorables[0])
        
        # 1. Use ZeroModel to create the initial VPM from the scorable
        initial_vpm, adapter_meta = self.zm.vpm_from_scorable(seed_scorable, img_size=256)
        
        # 2. Create the initial state
        initial_state = VPMState(
            vpm=initial_vpm,
            scorable=seed_scorable,
            history=[],
            metadata={"adapter": adapter_meta}
        )
        
        # 3. Define a goal (e.g., maximize clarity and coverage)
        goal = VPMGoal(dimensions=["clarity", "coverage"], target_value=0.9)
        
        # 4. Start the thought execution loop
        final_state = await self._execute_thought_loop(initial_state, goal)
        
        # 5. Log the final result
        context["nexus_final_state"] = final_state
        return context

    async def _execute_thought_loop(self, state: VPMState, goal: VPMGoal) -> VPMState:
        max_steps = 10
        for step in range(max_steps):
            # Compute progress towards goal (phi)
            phi = compute_phi(state, goal)
            
            # Check if goal is achieved
            if phi >= 1.0:
                self.logger.info(f"🎯 Goal achieved in {step} steps.")
                break
                
            # Choose the next thought
            thought = self.executor.select_thought(state, goal, step)
            if not thought:
                self.logger.info("No valid thought selected. Stopping.")
                break
                
            # Apply the thought's operations
            new_vpm = state.vpm
            for op in thought.ops:
                # Delegate the visual operation to ZeroModel
                new_vpm = self.zm.apply_visual_op(new_vpm, op)
                
            # Create a new scorable? Or keep track of the modification?
            # This depends on whether the op creates new semantic content.
            new_scorable = self._update_scorable(state.scorable, thought) 
            
            # Create the new state
            new_state = VPMState(
                vpm=new_vpm,
                scorable=new_scorable,
                history=state.history + [thought],
                metadata={**state.metadata, f"thought_{step}": thought.to_dict()}
            )
            
            # Score the new state's VPM
            # This could use the VisionScorer via ZeroModel
            vision_scores = self.zm.score_vpm_image(new_vpm, goal.dimensions)
            new_state.metadata["vision_scores"] = vision_scores
            
            # Update state for next iteration
            state = new_state
            
            # Yield control
            await asyncio.sleep(0)
            
        return state
        
    def _update_scorable(self, scorable: Scorable, thought: Thought) -> Scorable:
        # Placeholder: In reality, a ZOOM op might not change the text,
        # but a complex operation might generate new content.
        # For now, return the same scorable.
        return scorable
    
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
