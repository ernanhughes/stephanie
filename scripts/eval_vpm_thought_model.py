# stripts/eval_vpm_thought_model.py
from __future__ import annotations

import argparse
import json
import numpy as np 
import torch 
import networkx as nx
from omegaconf import OmegaConf
from stephanie.services.graph_layout import render_multi_layout_vpm
from stephanie.zeromodel.state_machine import VPMGoal, VPMState, compute_phi, Thought, ThoughtExecutor
from stephanie.components.nexus.utils.visual_thought import VisualThoughtOp, VisualThoughtType
from train_vpm_thought_model import VPMThoughtModel, DEFAULT_CONFIG  # reuse defs

def load_model(ckpt_path: str, device: str):
    cfg = OmegaConf.create(DEFAULT_CONFIG)
    model = VPMThoughtModel(cfg)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    return model, cfg

def sample_probe(seed=0):
    G = nx.barbell_graph(15, 4)  # hard case
    return "barbell", G

def eval_once(model, cfg, device="cpu"):
    ptype, G = sample_probe()
    vpms, metas = render_multi_layout_vpm(G, layouts=["forceatlas2"], config={"img_size":256})
    vpm0 = vpms[0].astype(np.float32)/255.0
    meta0 = metas[0]
    goal = VPMGoal(weights={"bridge_proxy": -1.0, "separability": 1.0})
    state = VPMState(X=vpm0, meta=meta0, phi=compute_phi(vpm0, meta0), goal=goal)

    # Model proposes op + params
    C = int(cfg.model.in_channels)
    vpm_tensor = torch.from_numpy(vpm0).unsqueeze(0).to(device)  # [1,C,H,W]
    if vpm_tensor.shape[1] != C:  # replicate if needed
        vpm_tensor = vpm_tensor[:, :1].repeat(1, C, 1, 1)
    goal_vec = np.array([goal.weights.get(k,0.0) for k in ["separability","bridge_proxy","symmetry","spectral_gap"]], dtype=np.float32)
    goal_tensor = torch.from_numpy(goal_vec).unsqueeze(0).to(device)

    with torch.no_grad():
        op_logits, param_mean, _, _ = model(vpm_tensor, goal_tensor)
        op_idx = int(op_logits.argmax(-1).item())
        pm = param_mean[0].cpu().numpy()

    # Decode to visual thought (simple mapping)
    ops = ["zoom","bbox","path","highlight","blur"]
    op = ops[op_idx]
    cx = int(np.clip(pm[0], -1, 1) * 127.5 + 127.5)
    cy = int(np.clip(pm[1], -1, 1) * 127.5 + 127.5)
    scale = float(np.clip(pm[2], 0.0, 1.0) * 4.0)
    vt = VisualThoughtOp(VisualThoughtType(op), {"center": (cx,cy), "scale": max(1.0, scale)}) if op=="zoom" else \
         VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (128,128), "scale": 2.0})

    execu = ThoughtExecutor(visual_op_cost={"zoom":1.0, "bbox":0.3,"path":0.4,"highlight":0.5,"blur":0.6})
    new_state, delta, cost, _ = execu.score_thought(state, Thought(name="EvalOp", ops=[vt], cost=1.0))

    return {"probe": ptype, "op": op, "delta": float(delta), "cost": float(cost), "center": (cx,cy), "scale": scale}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, args.device)
    reports = [eval_once(model, cfg, args.device) for _ in range(10)]
    mean_delta = float(np.mean([r["delta"] for r in reports]))
    succ_rate = float(np.mean([1.0 if r["delta"] > 0 else 0.0 for r in reports]))
    ops_hist = {}
    for r in reports: ops_hist[r["op"]] = ops_hist.get(r["op"],0)+1

    out = {
        "ckpt": args.ckpt,
        "mean_delta": mean_delta,
        "success_rate": succ_rate,
        "ops_hist": ops_hist,
        "samples": reports[:3],
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
