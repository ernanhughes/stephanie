# stephanie/components/nexus/scripts/rollout_vpm_thought.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as iio
import networkx as nx
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from stephanie.components.nexus.graph.graph_layout import \
    render_multi_layout_vpm
from stephanie.components.nexus.tools.train_vpm_thought_model import (  # Reuse model definition
    DEFAULT_CONFIG, VPMThoughtModel)
from stephanie.components.nexus.utils.visual_thought import (VisualThoughtOp,
                                                             VisualThoughtType)
from stephanie.components.nexus.vpm.state_machine import (Thought,
                                                          ThoughtExecutor,
                                                          VPMGoal, VPMState,
                                                          compute_phi)


# --------------------------- Helpers ---------------------------
def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[VPMThoughtModel, DictConfig]:
    """Load trained model and config from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = OmegaConf.create(ckpt.get("config", DEFAULT_CONFIG))
    model = VPMThoughtModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg

def generate_probe(probe_type: str, seed: int = 0) -> Tuple[nx.Graph, List[List[int]]]:
    """Generate standard probe graphs matching training data."""
    if probe_type == "barbell":
        G = nx.barbell_graph(15, 4)
        comms = [list(range(0, 15)), list(range(19, 34))]
    elif probe_type == "ring_of_cliques":
        G = nx.ring_of_cliques(4, 8)
        comms = [list(range(i*8, (i+1)*8)) for i in range(4)]
    elif probe_type == "sbm":
        blocks = (20, 20, 20)
        p_in, p_out = 0.3, 0.01
        probs = [[p_in if i==j else p_out for j in range(len(blocks))] for i in range(len(blocks))]
        G = nx.stochastic_block_model(blocks, probs, seed=seed)
        comms = [list(range(0, 20)), list(range(20, 40)), list(range(40, 60))]
    elif probe_type == "grid_maze":
        G = nx.grid_2d_graph(10, 10)
        comms = [[(i, j) for i in range(10) for j in range(10)]]
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    return G, comms

def render_vpm_for_display(vpm: np.ndarray, title: str = "", figsize=(3,3)):
    """Render VPM channels as a composite RGB image for display."""
    # vpm: [C, H, W] - we'll use node density (0), edge density (1), degree heat (2)
    if vpm.shape[0] < 3:
        # Pad with zeros if not enough channels
        padded = np.zeros((3, vpm.shape[1], vpm.shape[2]), dtype=vpm.dtype)
        padded[:vpm.shape[0]] = vpm
        vpm = padded
    
    # Normalize each channel to [0, 255]
    img = np.zeros((vpm.shape[1], vpm.shape[2], 3), dtype=np.uint8)
    for i in range(3):
        ch = vpm[i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)  # Normalize
        ch = (ch * 255).astype(np.uint8)
        img[:, :, i] = ch
    
    # Convert to PIL for annotation
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Add title
    if title:
        draw.text((5, 5), title, fill=(255, 255, 255), font=None)
    
    return np.array(pil_img)

def apply_thought_and_render(
    state: VPMState,
    thought: Thought,
    executor: ThoughtExecutor,
    step_idx: int,
    output_dir: Path,
    device: str = "cpu"
) -> Tuple[VPMState, float, float, float, np.ndarray]:
    """Apply thought, compute metrics, and render state for filmstrip."""
    # Apply thought to get new state
    new_state, delta, cost, bcs = executor.score_thought(state, thought)
    
    # Render VPM for display
    img = render_vpm_for_display(new_state.X, title=f"Step {step_idx}: {thought.name}\nΔ={delta:.3f}, BCS={bcs:.3f}")
    
    # Save individual frame
    frame_path = output_dir / f"frame_{step_idx:02d}.png"
    Image.fromarray(img).save(frame_path)
    
    return new_state, delta, cost, bcs, img

# --------------------------- Main rollout ---------------------------
def rollout_vpm_thought(
    checkpoint_path: str,
    probe_type: str = "barbell",
    max_steps: int = 3,
    output_dir: str = "rollouts",
    device: str = "cpu",
    seed: int = 42
):
    """Perform a visual thought rollout and generate filmstrip."""
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    output_dir = Path(output_dir) / f"{probe_type}_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"🎬 Starting rollout for {probe_type} (seed={seed})")
    print(f"💾 Saving to: {output_dir.absolute()}")
    
    # 1. Load model
    model, cfg = load_model(checkpoint_path, device)
    print(f"✅ Loaded model from {checkpoint_path}")
    
    # 2. Generate probe graph
    G, comms = generate_probe(probe_type, seed)
    print(f"📊 Generated {probe_type} graph with {G.number_of_nodes()} nodes")
    
    # 3. Render initial VPM
    vpms, metas = render_multi_layout_vpm(
        G,
        layouts=["forceatlas2"],
        config={"img_size": 256, "cache_dir": ".rollout_cache"}
    )
    initial_vpm = vpms[0].astype(np.float32) / 255.0  # [C, H, W]
    initial_meta = metas[0]
    
    # 4. Determine goal based on probe type
    if probe_type == "barbell":
        goal = VPMGoal(weights={"bridge_proxy": -1.0, "separability": 1.0}, task_type="bottleneck_detection")
    elif probe_type in ["sbm", "ring_of_cliques"]:
        goal = VPMGoal(weights={"symmetry": 1.0, "separability": 1.0}, task_type="community_analysis")
    else:
        goal = VPMGoal(weights={"separability": 1.0}, task_type="generic")
    
    # 5. Initialize state
    initial_phi = compute_phi(initial_vpm, initial_meta)
    state = VPMState(
        X=initial_vpm,
        meta=initial_meta,
        phi=initial_phi,
        goal=goal
    )
    initial_utility = state.utility
    print(f"🎯 Initial utility: {initial_utility:.4f}")
    print(f"   Metrics: separability={initial_phi.get('separability',0):.2f}, bridge_proxy={initial_phi.get('bridge_proxy',0):.2f}")
    
    # 6. Setup executor
    executor = ThoughtExecutor(
        visual_op_cost={"zoom": 1.0, "bbox": 0.3, "path": 0.4, "highlight": 0.5, "blur": 0.6}
    )
    
    # 7. Render initial state
    initial_img = render_vpm_for_display(initial_vpm, title=f"Initial State\nUtility={initial_utility:.4f}")
    Image.fromarray(initial_img).save(output_dir / "frame_00.png")
    frames = [initial_img]
    
    # 8. Rollout loop
    trajectory = []
    total_utility_gain = 0.0
    
    for step in range(1, max_steps + 1):
        print(f"\n🔍 Step {step}/{max_steps}")
        
        # Prepare inputs for model
        goal_vec = np.array([
            goal.weights.get("separability", 0.0),
            goal.weights.get("bridge_proxy", 0.0),
            goal.weights.get("symmetry", 0.0),
            goal.weights.get("spectral_gap", 0.0)
        ], dtype=np.float32)
        
        # Convert to tensors
        vpm_tensor = torch.from_numpy(state.X).float().unsqueeze(0).to(device)  # [1, C, H, W]
        
        # Ensure correct channel count (replicate if needed)
        if vpm_tensor.shape[1] != cfg.model.in_channels:
            vpm_tensor = vpm_tensor[:, :1].repeat(1, cfg.model.in_channels, 1, 1)
        
        goal_tensor = torch.from_numpy(goal_vec).float().unsqueeze(0).to(device)  # [1, 4]
        
        # Get model prediction
        with torch.no_grad():
            op_logits, param_mean, _, value_pred = model(vpm_tensor, goal_tensor)
        
        # Sample action (greedy)
        op_idx = int(torch.argmax(op_logits, dim=1).item())
        params = param_mean[0].cpu().numpy()
        
        # Map to op name
        op_names = ["zoom", "bbox", "path", "highlight", "blur"]
        op_name = op_names[op_idx] if op_idx < len(op_names) else "zoom"
        
        # Decode parameters to thought
        if op_name == "zoom":
            # params[0], params[1] = normalized center coordinates
            cx = int(np.clip(params[0], -1, 1) * 127.5 + 127.5)
            cy = int(np.clip(params[1], -1, 1) * 127.5 + 127.5)
            scale = float(np.clip(params[2], 0.0, 1.0) * 4.0 + 1.0)  # 1.0 to 5.0
            thought = Thought(
                name=f"Zoom_{step}",
                ops=[VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (cx, cy), "scale": scale})],
                intent=f"Zoom on region with high {goal.task_type}",
                cost=1.0
            )
        elif op_name == "bbox":
            # params[0-3] = normalized bbox coordinates
            x1 = int(np.clip(params[0], -1, 1) * 127.5 + 127.5)
            y1 = int(np.clip(params[1], -1, 1) * 127.5 + 127.5)
            x2 = int(np.clip(params[2], -1, 1) * 127.5 + 127.5)
            y2 = int(np.clip(params[3], -1, 1) * 127.5 + 127.5)
            thought = Thought(
                name=f"Box_{step}",
                ops=[VisualThoughtOp(VisualThoughtType.BBOX, {"xyxy": (x1, y1, x2, y2)})],
                intent=f"Highlight region of interest for {goal.task_type}",
                cost=0.3
            )
        else:
            # Fallback to zoom for other ops (simplified)
            thought = Thought(
                name=f"DefaultZoom_{step}",
                ops=[VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (128, 128), "scale": 2.0})],
                cost=1.0
            )
        
        print(f"💡 Model proposed: {thought.name} ({op_name})")
        print(f"   Parameters: {thought.ops[0].params}")
        
        # Apply thought and render
        new_state, delta, cost, bcs, img = apply_thought_and_render(
            state, thought, executor, step, output_dir, device
        )
        
        # Record step
        trajectory.append({
            "step": step,
            "thought": thought.name,
            "op": op_name,
            "params": thought.ops[0].params,
            "delta": float(delta),
            "cost": float(cost),
            "bcs": float(bcs),
            "utility": float(new_state.utility),
            "value_pred": float(value_pred.item())
        })
        
        print(f"   Δutility: {delta:.4f}, BCS: {bcs:.4f}, Cost: {cost:.2f}")
        print(f"   New utility: {new_state.utility:.4f}")
        
        # Update state
        state = new_state
        total_utility_gain += delta
        frames.append(img)
        
        # Early stopping if no gain
        if delta < 0.01:
            print("   ⚠️  Stopping early: no significant utility gain")
            break
    
    # 9. Save filmstrip GIF
    gif_path = output_dir / "filmstrip.gif"
    iio.mimsave(gif_path, frames, fps=1, loop=0)
    print(f"\n✅ Filmstrip saved to: {gif_path.absolute()}")
    
    # 10. Save trajectory metrics
    metrics = {
        "probe_type": probe_type,
        "seed": seed,
        "initial_utility": float(initial_utility),
        "final_utility": float(state.utility),
        "total_delta": float(total_utility_gain),
        "total_steps": len(trajectory),
        "trajectory": trajectory,
        "model_checkpoint": checkpoint_path
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved to: {output_dir / 'metrics.json'}")
    
    # 11. Print summary
    print("\n📈 Rollout Summary:")
    print(f"Initial utility: {initial_utility:.4f}")
    for step in trajectory:
        print(f"Step {step['step']}: {step['thought']} → Δ={step['delta']:.4f}, BCS={step['bcs']:.4f}")
    print(f"Final utility: {state.utility:.4f}")
    print(f"Total utility gain: {total_utility_gain:.4f} ({(total_utility_gain/initial_utility)*100:.1f}% improvement)")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout VPM Thought Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--probe", type=str, default="barbell", choices=["barbell", "sbm", "ring_of_cliques", "grid_maze"], help="Probe type")
    parser.add_argument("--steps", type=int, default=3, help="Max steps to rollout")
    parser.add_argument("--output", type=str, default="rollouts", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    rollout_vpm_thought(
        checkpoint_path=args.checkpoint,
        probe_type=args.probe,
        max_steps=args.steps,
        output_dir=args.output,
        device=args.device,
        seed=args.seed
    )
