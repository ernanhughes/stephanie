# stephanie/components/nexus/tools/train_vpm_thought_model.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx  # keep early import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from stephanie.components.nexus.graph.graph_layout import \
    render_multi_layout_vpm
from stephanie.components.nexus.utils.visual_thought import (VisualThoughtOp,
                                                             VisualThoughtType)
from stephanie.components.nexus.vpm.state_machine import (Thought,
                                                          ThoughtExecutor,
                                                          VPMGoal, VPMState,
                                                          compute_phi)

DEFAULT_CONFIG = {
    "model": {
        "hidden_dim": 512,
        "goal_dim": 128,
        "n_ops": 5,          # zoom, bbox, path, highlight, blur
        "param_dim": 8,      # x,y,scale,width,height,arrow,etc.
        "dropout": 0.1,
        "in_channels": 3,
    },
    "training": {
        "batch_size": 16,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "max_epochs": 12,
        "clip_grad": 1.0,
        "device": "cpu",
        "seed": 42,
        "num_workers": 4,
    },
    "curriculum": {
        "steps": [1, 2, 3],
        "samples_per_step": [2000, 1500, 1000],
        "min_utility_gain": [0.05, 0.03, 0.01],
    },
    "data": {
        "probe_types": ["sbm", "ring_of_cliques", "barbell", "grid_maze"],
        "real_traces_sample": 0.0,
        "max_traces": 5000,
    },
    "rewards": {
        "utility_weight": 1.0,
        "format_weight": 0.5,
        "risk_weight": 0.3,
        "cost_weight": -0.2,
        "sicql_weight": 0.0,   # set >0 when you pass sicql_q in batch
    },
    "evaluation": {
        "save_every": 1,   # save every epoch
        "log_every": 50,
    },
    "paths": {
        "models_dir": "models/vpm_thought",
        "logs_dir": "logs/vpm_thought",
    },
    "resume_from": None,
}

# ---------------- Model ----------------
class VPMSpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1), nn.ReLU(),       # 256->128
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),               # 128->64
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),              # 64->32
            nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),              # 32->16
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.proj = nn.Sequential(nn.Linear(512, out_dim), nn.LayerNorm(out_dim), nn.ReLU())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))

class VPMThoughtModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        C = int(config.model.in_channels)
        H = int(config.model.hidden_dim)
        G = int(config.model.goal_dim)
        self.encoder = VPMSpatialEncoder(C, H)
        self.goal_proj = nn.Sequential(nn.Linear(C, G), nn.ReLU(), nn.Linear(G, G))
        self.fuser = nn.Sequential(nn.Linear(H + G, H), nn.ReLU(), nn.Dropout(config.model.dropout))
        self.op_head = nn.Linear(H, int(config.model.n_ops))
        self.param_head = nn.Linear(H, int(config.model.param_dim) * 2)
        self.value_head = nn.Linear(H, 1)

    def forward(self, vpm: torch.Tensor, goal_vec: torch.Tensor):
        s = self.encoder(vpm)               # [B,H]
        g = self.goal_proj(goal_vec)        # [B,G]
        h = self.fuser(torch.cat([s, g], -1))
        op_logits = self.op_head(h)         # [B,n_ops]
        pr = self.param_head(h)             # [B,2*param_dim]
        D = int(self.config.model.param_dim)
        param_mean = torch.tanh(pr[:, :D])  # [-1,1]
        param_log_std = pr[:, D:]
        value = self.value_head(h)          # [B,1]
        return op_logits, param_mean, param_log_std, value

# ------------- Dataset (Iterable) -------------
class VPMThoughtDataset(IterableDataset):
    def __init__(self, config: DictConfig, curriculum_step: int = 1, executor: Optional[ThoughtExecutor] = None):
        self.config = config
        self.curriculum_step = curriculum_step
        self.executor = executor or ThoughtExecutor(
            visual_op_cost={"zoom":1.0, "bbox":0.3, "path":0.4, "highlight":0.5, "blur":0.6}
        )
        self.rng = np.random.RandomState(int(config.training.seed))
        self.probe_graphs = self._generate_probes()

    def _generate_probes(self):
        probes = []
        for ptype in self.config.data.probe_types:
            for i in range(200):
                if ptype == "sbm":
                    blocks = (20,20,20); p_in, p_out = 0.3, 0.01
                    probs = [[p_in if a==b else p_out for b in range(3)] for a in range(3)]
                    G = nx.stochastic_block_model(blocks, probs, seed=i)
                    comms = [list(range(0,20)), list(range(20,40)), list(range(40,60))]
                elif ptype == "ring_of_cliques":
                    G = nx.ring_of_cliques(4,8)
                    comms = [list(range(k*8, (k+1)*8)) for k in range(4)]
                elif ptype == "barbell":
                    G = nx.barbell_graph(15,4)
                    comms = [list(range(0,15)), list(range(19,34))]
                elif ptype == "grid_maze":
                    G = nx.grid_2d_graph(10,10)
                    comms = [[(i,j) for i in range(10) for j in range(10)]]
                else:
                    continue
                probes.append((ptype, G, comms))
        return probes

    def _get_goal_for_probe(self, ptype: str) -> VPMGoal:
        if ptype == "barbell":
            return VPMGoal(weights={"bridge_proxy": -1.0, "separability": 1.0})
        if ptype in ["sbm", "ring_of_cliques"]:
            return VPMGoal(weights={"symmetry": 1.0, "separability": 1.0})
        if ptype == "grid_maze":
            return VPMGoal(weights={"path_coherence": 1.0, "symmetry": -0.5})
        return VPMGoal(weights={"separability": 1.0})

    def _sample_random_thought(self, state: VPMState, goal: VPMGoal) -> Optional[Thought]:
        # Minimal valid sampler (center zoom). Replace with learned policy at rollout.
        return Thought(
            name="RandomZoom",
            ops=[VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (128,128), "scale": 2.0})],
            cost=1.0
        )

    def _encode_op(self, thought: Thought) -> Dict[str, Any]:
        op_map = {"zoom":0, "bbox":1, "path":2, "highlight":3, "blur":4}
        params = np.zeros(int(self.config.model.param_dim), dtype=np.float32)
        if thought.ops and thought.ops[0].type.value in op_map:
            op_type = op_map[thought.ops[0].type.value]
            p = thought.ops[0].params
            if "center" in p:
                cx, cy = p["center"]; params[0] = cx/255.0; params[1] = cy/255.0
            if "scale" in p:
                params[2] = min(4.0, float(p["scale"])) / 4.0
        else:
            op_type = 0
        return {"op_type": int(op_type), "params": params}

    def __iter__(self):
        rng = self.rng
        while True:
            ptype, G, comms = rng.choice(self.probe_graphs)
            goal = self._get_goal_for_probe(ptype)
            # Render initial VPM (single layout for speed)
            vpms, metas = render_multi_layout_vpm(G, layouts=["forceatlas2"], config={"img_size":256})
            vpm0 = vpms[0].astype(np.float32) / 255.0       # [C,H,W]
            meta0 = metas[0]
            state = VPMState(X=vpm0, meta=meta0, phi=compute_phi(vpm0, meta0), goal=goal)
            traj = []
            total_gain = 0.0
            min_gain = float(self.config.curriculum.min_utility_gain[self.curriculum_step-1])

            for _ in range(int(self.curriculum_step)):
                thought = self._sample_random_thought(state, goal)
                if not thought: break
                new_state, delta, cost, _ = self.executor.score_thought(state, thought)
                if delta < min_gain: break
                enc = self._encode_op(thought)
                goal_vec = np.array([goal.weights.get(k, 0.0) for k in ["separability","bridge_proxy","symmetry","spectral_gap"]], dtype=np.float32)
                traj.append({
                    "vpm": state.X.copy(),
                    "goal_vec": goal_vec,
                    "op": enc["op_type"],
                    "params": enc["params"],
                    "delta": float(delta),
                    "cost": float(cost),
                })
                state = new_state
                total_gain += float(delta)

            if total_gain > 0 and traj:
                for step in traj:
                    yield {
                        "vpm": torch.from_numpy(step["vpm"]).float(),            # [C,H,W]
                        "goal_vec": torch.from_numpy(step["goal_vec"]).float(),  # [C_goal]
                        "op": torch.tensor(step["op"], dtype=torch.long),
                        "params": torch.from_numpy(step["params"]).float(),
                        "delta": torch.tensor(step["delta"], dtype=torch.float32),
                        "cost": torch.tensor(step["cost"], dtype=torch.float32),
                    }

# ------------- Trainer -------------
class GRPOTrainer:
    def __init__(self, model: nn.Module, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        self.global_step = 0
        self.log_dir = Path(cfg.paths.logs_dir); self.log_dir.mkdir(parents=True, exist_ok=True)

    def compute_reward(self, batch: Dict[str, torch.Tensor], params: torch.Tensor) -> torch.Tensor:
        r = torch.zeros(batch["delta"].shape[0], device=self.device)
        r += float(self.cfg.rewards.utility_weight) * batch["delta"].to(self.device)
        param_valid = 1.0 - torch.mean((torch.abs(params) > 1.0).float(), dim=1)
        r += float(self.cfg.rewards.format_weight) * param_valid
        r += float(self.cfg.rewards.cost_weight) * batch["cost"].to(self.device)
        if "sicql_q" in batch and float(self.cfg.rewards.get("sicql_weight", 0.0)) > 0:
            r += float(self.cfg.rewards.sicql_weight) * batch["sicql_q"].to(self.device)
        return r

    def train_epoch(self, dl: DataLoader, epoch_idx: int):
        self.model.train()
        running = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch_idx+1}/{self.cfg.training.max_epochs}")
        for batch in pbar:
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
            # Simple channel sim: replicate to 3 channels
            vpm = batch["vpm"].unsqueeze(1).repeat(1, int(self.cfg.model.in_channels), 1, 1)
            op_logits, param_mean, param_log_std, value_pred = self.model(vpm, batch["goal_vec"])
            std = torch.exp(0.5 * param_log_std)
            params_sampled = param_mean + torch.randn_like(std) * std

            rewards = self.compute_reward(batch, params_sampled)
            adv = rewards - rewards.mean()
            if rewards.std() > 1e-6: adv = adv / (rewards.std() + 1e-8)

            op_loss = F.cross_entropy(op_logits, batch["op"])
            param_loss = F.mse_loss(params_sampled, batch["params"])
            value_loss = F.mse_loss(value_pred.squeeze(-1), rewards)

            log_prob_op = -F.cross_entropy(op_logits, batch["op"], reduction="none")
            log_prob_param = -0.5 * ((params_sampled - batch["params"]) ** 2).sum(dim=1)
            policy_loss = -(log_prob_op + 0.1 * log_prob_param) * adv.detach()
            policy_loss = policy_loss.mean()

            total = policy_loss + 0.5*value_loss + 0.1*op_loss + 0.01*param_loss

            self.optimizer.zero_grad()
            total.backward()
            if float(self.cfg.training.clip_grad) > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.training.clip_grad))
            self.optimizer.step()

            running += float(total.item())
            self.global_step += 1
            if self.global_step % int(self.cfg.evaluation.log_every) == 0:
                metrics = {
                    "epoch": epoch_idx, "step": self.global_step,
                    "loss": float(total.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "reward_mean": float(rewards.mean().item()),
                }
                with open(self.log_dir / f"step_{self.global_step}.json", "w") as f:
                    json.dump(metrics, f)
            pbar.set_postfix(loss=float(total.item()))
        return running / max(1,len(dl))

# ------------- Orchestration -------------
def save_ckpt(path: Path, model: nn.Module, curriculum_idx: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "curriculum_idx": curriculum_idx}, path)

def train(cfg: DictConfig):
    torch.manual_seed(int(cfg.training.seed)); np.random.seed(int(cfg.training.seed)); random.seed(int(cfg.training.seed))
    cfg.paths["models_dir"] = str(Path(cfg.paths.models_dir).expanduser())
    cfg.paths["logs_dir"] = str(Path(cfg.paths.logs_dir).expanduser())
    models_dir = Path(cfg.paths.models_dir); models_dir.mkdir(parents=True, exist_ok=True)

    model = VPMThoughtModel(cfg)
    start_stage = 0
    if cfg.get("resume_from"):
        ck = Path(cfg.resume_from)
        if ck.exists():
            print(f"📥 Resuming from {ck}")
            state = torch.load(ck, map_location=cfg.training.device)
            model.load_state_dict(state["model_state"])
            start_stage = int(state.get("curriculum_idx", 0))

    trainer = GRPOTrainer(model, cfg)

    total_stages = len(cfg.curriculum.steps)
    for stage_idx in range(start_stage, total_stages):
        n_steps = int(cfg.curriculum.steps[stage_idx])
        print(f"\n🚀 Curriculum stage {stage_idx+1}/{total_stages}: {n_steps}-step thoughts")
        ds = VPMThoughtDataset(cfg, curriculum_step=n_steps)
        dl = DataLoader(ds, batch_size=int(cfg.training.batch_size), num_workers=int(cfg.training.num_workers))
        # Split epochs across stages
        epochs_this_stage = max(1, cfg.training.max_epochs // total_stages)
        for epoch in range(epochs_this_stage):
            loss = trainer.train_epoch(dl, epoch)
            print(f"  epoch {epoch+1}/{epochs_this_stage} loss={loss:.4f}")
            if (epoch+1) % int(cfg.evaluation.save_every) == 0:
                save_ckpt(models_dir / f"vpm_thought_stage{stage_idx}_epoch{epoch+1}.pt", model, stage_idx)
        # stage checkpoint
        save_ckpt(models_dir / f"vpm_thought_stage{stage_idx}_final.pt", model, stage_idx)

    # final
    save_ckpt(models_dir / "vpm_thought_final.pt", model, total_stages-1)
    print(f"✅ Done. Final model at: {models_dir/'vpm_thought_final.pt'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config (optional)")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    # Load user config or use defaults
    base = OmegaConf.create(DEFAULT_CONFIG)
    if args.config:
        user = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(base, user)
    else:
        cfg = base
    if args.resume_from:
        cfg.resume_from = args.resume_from

    train(cfg)

if __name__ == "__main__":
    main()
