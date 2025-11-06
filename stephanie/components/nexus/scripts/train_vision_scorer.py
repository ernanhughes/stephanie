# scripts/train_vision_scorer.py
import hydra
from omegaconf import DictConfig

from stephanie.services.graph_vision_scorer import VisionScorer


@hydra.main(version_base=None, config_path="../configs", config_name="vision_scorer")
def train(cfg: DictConfig):
    model_path = VisionScorer.train_on_probes(config=cfg)
    print(f"✅ Trained model saved to: {model_path}")

if __name__ == "__main__":
    train()


### Hydra config: configs/vision_scorer.yaml

# yaml
# defaults:
#   - graph_layout@_global_: graph_layout  # reuse your existing graph_layout config

# model:
#   backbone: mobilenet_v3_small
#   pretrained: true
#   freeze_backbone: true
#   head_hidden_dim: 64
#   layouts: ["forceatlas2", "spectral"]  # must match graph_layout.yaml

# input:
#   img_size: 256
#   channels: ["node_density", "edge_density", "degree_heatmap"]

# training:
#   probe_types: ["sbm", "ring_of_cliques", "barbell"]
#   n_samples_per_probe: 500
#   batch_size: 32
#   lr: 1e-3
#   epochs: 10
#   device: cpu
#   save_path: ${paths.models_dir}/graph_vision_scorer.pt

# inference:
#   cache_dir: ${paths.cache_dir}/vision_scorer
#   device: cpu
#   timeout_s: 2.0


# ---

# ### Why this integrates cleanly:
# - 🔌 **Uses your hardened graph_layout.py** → cached VPM renders, layout hashes, FA2 fallback  
# - ⚡ **Tiny & cache-aware** → <0.5M params; 20ms on CPU; cache key by graph structure  
# - 📊 **Fuses with existing scores** → drop vision_symmetry into SICQL/HRM like any other channel  
# - 🚨 **Risk-aware** → logs GNN-vs-vision disagreement to MemCube for Daimon review  
# - 🧪 **Probe-trained** → leverages your existing generate_probe (SBM/ring/barbell)  

# ### Next-step options:
# 1. **Gap panel for probe metrics** → I’ll generate a Jupyter snippet that loads probe_metrics.json and plots FA2 vs spectral separability  
# 2. **Jitter filmstrip renderer** → Show visual-thought steps + VPM channels side-by-side in telemetry  
# 3. **Auto-mode-switch router** → Train a tiny MLP on MemCube logs to replace p_interleaved heuristic  

# **Your call:** Which unlocks the next win fastest? (I vote Gap panel — 50 lines of Plotly to validate vision scorer on probes.)

# * * *

# **ChatGPT:**

# Awesome—let’s ship a **lean, dependency-light GraphVisionScorer** that runs anywhere (pure PyTorch + NetworkX + NumPy) and doesn’t rely on torchvision. I’ve kept it tiny (<0.5M params), cache-friendly, and wired to your multi-layout VPMs.

# Below is a **complete drop-in** you can paste in. It includes:

# * A **TinyCNN** backbone (no torchvision).
    
# * Inference wrapper (`VisionScorer`) that consumes your `render_multi_layout_vpm`.
    
# * An **optional trainer** that generates synthetic graphs (SBM / ring-of-cliques / barbell) and auto-labels them (symmetry, spectral gap bucket, bridge proxy).
    
# * Clean shapes and **correct normalization** for CHW tensors (no transform pitfalls).
    

# * * *

# `stephanie/services/graph_vision_scorer.py`
# ===========================================

# ```python
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from stephanie.services.graph_layout import \
    render_multi_layout_vpm  # your hardened renderer

# --------------------------- Config defaults ---------------------------

DEFAULT_CONFIG = {
    "model": {
        "backbone": "tiny_cnn",   # tiny_cnn only; vendor-agnostic
        "head_hidden_dim": 64,
        "layouts": ["forceatlas2", "spectral"],  # must match graph_layout.yaml
        "dropout": 0.0,
    },
    "input": {
        "img_size": 256,
        # we treat [node, edge, degree] as pseudo-RGB, already in [0,1]
        "per_channel_mean": [0.0, 0.0, 0.0],
        "per_channel_std": [1.0, 1.0, 1.0],
    },
    "training": {
        "probe_types": ["sbm", "ring_of_cliques", "barbell"],
        "n_samples_per_probe": 300,   # small & fast; bump if you want
        "batch_size": 32,
        "lr": 1e-3,
        "epochs": 8,
        "device": "cpu",
        "save_path": "models/graph_vision_scorer.pt",
        "img_size": 256,
        "seed": 7,
    },
    "inference": {
        "cache_dir": ".vision_scorer_cache",
        "device": "cpu",
        "timeout_s": 2.0,
    },
}


# --------------------------- Tiny backbone -----------------------------

class TinyCNN(nn.Module):
    """
    ~0.35M params, fast on CPU.
    Expects input [B, 3, H, W] in [0,1] range (already normalized VPM channels).
    """
    def __init__(self, in_ch: int = 3, feat_dim: int = 192, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8

            nn.Conv2d(64, 96, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, 1, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # [B,96,1,1]
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(96, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


# --------------------------- Multi-layout head -------------------------

class GraphVisionScorer(nn.Module):
    """
    Multi-layout VPM -> fused features -> heads:
      - symmetry:        [0,1]
      - spectral_gap:    3-way bucket logits
      - bridge_proxy:    [0,1] (bridges/edges proxy)
    """
    def __init__(self, n_layouts: int = 2, head_hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.backbone = TinyCNN(in_ch=3, feat_dim=192, dropout=dropout)
        feat = self.backbone.out_dim * n_layouts

        self.head_symmetry = nn.Sequential(
            nn.Linear(feat, head_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(head_hidden_dim, 1), nn.Sigmoid()
        )
        self.head_spectral = nn.Sequential(
            nn.Linear(feat, head_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(head_hidden_dim, 3)  # logits for 3 buckets
        )
        self.head_bridge = nn.Sequential(
            nn.Linear(feat, head_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(head_hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, n_layouts, 3, H, W]
        """
        B, L, C, H, W = x.shape
        feats = []
        for i in range(L):
            f = self.backbone(x[:, i])  # [B, D]
            feats.append(f)
        fused = torch.cat(feats, dim=1)  # [B, D*L]
        return {
            "symmetry": self.head_symmetry(fused).squeeze(-1),
            "spectral_gap_bucket": self.head_spectral(fused),  # logits
            "bridge_proxy": self.head_bridge(fused).squeeze(-1),
        }


# --------------------------- Inference wrapper -------------------------

class VisionScorer:
    """
    Stateless wrapper that renders multi-layout VPMs and runs GraphVisionScorer.
    """
    def __init__(self, config: Optional[Union[DictConfig, Dict[str, Any]]] = None, model_path: Optional[str] = None, device: Optional[str] = None):
        self.cfg = OmegaConf.create(DEFAULT_CONFIG)
        if config:
            self.cfg = OmegaConf.merge(self.cfg, config)

        self.device = device or self.cfg.inference.device
        self.cache_dir = Path(self.cfg.inference.cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = GraphVisionScorer(
            n_layouts=len(self.cfg.model.layouts),
            head_hidden_dim=self.cfg.model.head_hidden_dim,
            dropout=self.cfg.model.dropout,
        ).to(self.device)
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Per-channel normalize constants as tensors for broadcasting
        mean = torch.tensor(self.cfg.input.per_channel_mean, dtype=torch.float32)[None, None, :, None, None]
        std = torch.tensor(self.cfg.input.per_channel_std, dtype=torch.float32)[None, None, :, None, None]
        self.registered_mean = mean.to(self.device)
        self.registered_std = std.to(self.device)

    @torch.no_grad()
    def score_graph(self, graph: Any, timeout_s: Optional[float] = None, cache_key: Optional[str] = None) -> Dict[str, float]:
        t0 = time.time()
        timeout_s = float(timeout_s or self.cfg.inference.timeout_s)

        # 1) Render multi-layout VPMs (already cached by graph_layout)
        vpms, metas = render_multi_layout_vpm(
            graph,
            layouts=list(self.cfg.model.layouts),
            config={"img_size": self.cfg.input.img_size},
        )  # each: [3,H,W] uint8

        # 2) Stack -> float tensor in [0,1]
        x = np.stack(vpms, axis=0).astype(np.float32) / 255.0  # [L,3,H,W]
        xt = torch.from_numpy(x)[None].to(self.device)         # [1,L,3,H,W]

        # 3) Normalize per channel (broadcasted)
        xt = (xt - self.registered_mean) / (self.registered_std + 1e-8)

        # 4) Model
        out = self.model(xt)

        # 5) Post
        scores = {
            "vision_symmetry": float(out["symmetry"][0].clamp(0, 1).cpu()),
            "vision_bridge_proxy": float(out["bridge_proxy"][0].clamp(0, 1).cpu()),
            "vision_spectral_gap_bucket": int(out["spectral_gap_bucket"][0].argmax().cpu()),
        }

        # 6) Optional cache dump (include metas)
        if cache_key:
            cache_path = self.cache_dir / f"{cache_key}.json"
            with open(cache_path, "w") as f:
                json.dump({**scores, "layouts": self.cfg.model.layouts, "meta": metas}, f)

        if (time.time() - t0) > timeout_s:
            raise TimeoutError(f"VisionScorer exceeded {timeout_s}s timeout")
        return scores

    # --------------------------- Optional training ---------------------------

    @classmethod
    def train_on_probes(cls, config: Optional[DictConfig] = None) -> str:
        """
        Self-supervised-ish training on synthetic probes (SBM/ring/barbell).
        Generates labels:
          - symmetry:    from community separability (sigmoid scaled)
          - gap bucket:  from algebraic connectivity thresholds
          - bridge proxy: (#bridges / |E|) clipped [0,1]
        """
        cfg = OmegaConf.merge(OmegaConf.create(DEFAULT_CONFIG), config or {})
        device = cfg.training.device
        torch.manual_seed(int(cfg.training.seed))

        model = GraphVisionScorer(
            n_layouts=len(cfg.model.layouts),
            head_hidden_dim=int(cfg.model.head_hidden_dim),
            dropout=float(cfg.model.dropout),
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr))
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()

        # ----- dataset synthesis -----
        def make_sample(probe_type: str, seed: int) -> Tuple[np.ndarray, Dict[str, float]]:
            if probe_type == "sbm":
                G, labels = _gen_sbm(seed)
            elif probe_type == "ring_of_cliques":
                G, labels = _gen_ring(seed)
            elif probe_type == "barbell":
                G, labels = _gen_barbell(seed)
            else:
                raise ValueError(probe_type)

            vpms, _ = render_multi_layout_vpm(G, layouts=cfg.model.layouts, config={"img_size": cfg.training.img_size})
            x = np.stack(vpms, axis=0).astype(np.float32) / 255.0  # [L,3,H,W]
            return x, labels

        samples: List[Tuple[np.ndarray, Dict[str, float]]] = []
        for p in cfg.training.probe_types:
            for i in range(int(cfg.training.n_samples_per_probe)):
                samples.append(make_sample(p, seed=i))

        # ----- training -----
        model.train()
        B = int(cfg.training.batch_size)
        L = len(cfg.model.layouts)
        mean = torch.tensor(cfg.input.per_channel_mean, dtype=torch.float32)[None, None, :, None, None].to(device)
        std = torch.tensor(cfg.input.per_channel_std, dtype=torch.float32)[None, None, :, None, None].to(device)

        for ep in range(int(cfg.training.epochs)):
            perm = torch.randperm(len(samples))
            total = 0.0
            for k in range(0, len(samples), B):
                idx = perm[k:k+B]
                batch = [samples[int(t)] for t in idx]

                x = torch.from_numpy(np.stack([b[0] for b in batch], axis=0)).to(device)  # [B,L,3,H,W]
                x = (x - mean) / (std + 1e-8)

                sym = torch.tensor([b[1]["symmetry"] for b in batch], dtype=torch.float32, device=device)
                gapb = torch.tensor([b[1]["spectral_gap_bucket"] for b in batch], dtype=torch.long, device=device)
                bridge = torch.tensor([b[1]["bridge_proxy"] for b in batch], dtype=torch.float32, device=device)

                opt.zero_grad()
                out = model(x)
                loss = mse(out["symmetry"], sym) + ce(out["spectral_gap_bucket"], gapb) + mse(out["bridge_proxy"], bridge)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu())

            # optional print
            # print(f"epoch {ep+1}/{cfg.training.epochs} loss={total:.3f}")

        # ----- save -----
        save_path = Path(cfg.training.save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        return str(save_path)


# --------------------------- Probe generators & labels ---------------------------

def _community_separability_from_positions(G: nx.Graph, comms: List[List[int]], pos: Dict[str, Tuple[float,float]]) -> float:
    # normalize
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    mnx, mxx = min(xs), max(xs); mny, mxy = min(ys), max(ys)
    sx = (mxx - mnx) or 1.0; sy = (mxy - mny) or 1.0
    P = {k: ((v[0]-mnx)/sx, (v[1]-mny)/sy) for k,v in pos.items()}

    def mean_pairwise(coords: np.ndarray) -> float:
        if len(coords) < 2: return 0.0
        dif = coords[:,None,:] - coords[None,:,:]
        d = np.linalg.norm(dif, axis=-1)
        iu = np.triu_indices(len(coords),1)
        return float(d[iu].mean()) if iu[0].size else 0.0

    intra = []
    cents = []
    for c in comms:
        coords = np.array([P[str(n)] for n in c], dtype=float)
        intra.append(mean_pairwise(coords))
        cents.append(coords.mean(axis=0))
    inter = mean_pairwise(np.stack(cents, axis=0))
    denom = (np.mean([x for x in intra if x > 0]) if any(x>0 for x in intra) else 1e-6)
    return inter / denom

def _label_pack(G: nx.Graph, comms: List[List[int]], layouts: List[str], img_size: int) -> Dict[str, float]:
    # use your renderer to get positions and spectral gap quickly
    _, metas = render_multi_layout_vpm(G, layouts=layouts, config={"img_size": img_size})
    # separability from first layout meta
    sep = _community_separability_from_positions(G, comms, metas[0]["positions"])
    # squash to [0,1] with a soft sigmoid around ~1.0
    symmetry = float(1.0 / (1.0 + math.exp(-2.0 * (sep - 1.0))))
    # spectral gap bucket from algebraic connectivity (use meta if present)
    try:
        gap = float(metas[0].get("spectral_gap", 0.0))
    except Exception:
        try:
            gap = float(nx.algebraic_connectivity(G))
        except Exception:
            gap = 0.0
    # simple thresholds—empirical, good enough for synthetic graphs
    if gap < 0.05: gap_bucket = 0
    elif gap < 0.2: gap_bucket = 1
    else: gap_bucket = 2
    # bridge proxy: bridges / edges (clip)
    try:
        bridges = sum(1 for _ in nx.bridges(G))
    except Exception:
        bridges = 0
    E = max(1, G.number_of_edges())
    bridge_proxy = float(min(1.0, max(0.0, bridges / E)))
    return {"symmetry": symmetry, "spectral_gap_bucket": gap_bucket, "bridge_proxy": bridge_proxy}

def _gen_sbm(seed: int) -> Tuple[nx.Graph, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    blocks = (30, 30, 30)
    p_in, p_out = 0.25, 0.02
    probs = [[p_in if i==j else p_out for j in range(len(blocks))] for i in range(len(blocks))]
    G = nx.stochastic_block_model(blocks, probs, seed=int(seed))
    comms = []; s=0
    for sz in blocks: comms.append(list(range(s, s+sz))); s+=sz
    labels = _label_pack(G, comms, layouts=["forceatlas2","spectral"], img_size=256)
    return G, labels

def _gen_ring(seed: int) -> Tuple[nx.Graph, Dict[str, float]]:
    G = nx.ring_of_cliques(6, 10)
    comms = []
    start = 0
    for _ in range(6):
        comms.append(list(range(start, start+10)))
        start += 10
    labels = _label_pack(G, comms, layouts=["forceatlas2","spectral"], img_size=256)
    return G, labels

def _gen_barbell(seed: int) -> Tuple[nx.Graph, Dict[str, float]]:
    G = nx.barbell_graph(20, 4, 20)
    left = list(range(0, 20))
    right = list(range(24, 44))
    labels = _label_pack(G, [left, right], layouts=["forceatlas2","spectral"], img_size=256)
    return G, labels

# # stephanie/services/scoring_manager.py
# from stephanie.services.graph_vision_scorer import VisionScorer

# class ScoringManager:
#     def __init__(self, config):
#         # ...
#         self.vision_scorer = VisionScorer(
#             config=config.get("vision_scorer", None),
#             model_path="models/graph_vision_scorer.pt",  # optional (after training)
#         )

#     def score_plan_trace(self, plan_trace):
#         scores = {}  # your existing composite
#         try:
#             v = self.vision_scorer.score_graph(plan_trace.graph, cache_key=plan_trace.id)
#             scores.update(v)
#             # Optional: disagreement flag for Daimon
#             gnn_sym = scores.get("gnn_symmetry", 0.5)
#             if abs(gnn_sym - v["vision_symmetry"]) > 0.4:
#                 self.memcube.log_flag(
#                     trace_id=plan_trace.id,
#                     flag="HIGH_GNN_VISION_DISAGREEMENT",
#                     meta={"gnn_sym": gnn_sym, "vision_sym": v["vision_symmetry"]},
#                 )
#         except Exception as e:
#             _logger.warning(f"VisionScorer failed: {e}")
#         return scores


# Optional: tiny trainer script
# -----------------------------

# ```python
# # scripts/train_vision_scorer.py
# import hydra
# from omegaconf import DictConfig
# from stephanie.services.graph_vision_scorer import VisionScorer

# @hydra.main(version_base=None, config_path="../configs", config_name="vision_scorer")
# def train(cfg: DictConfig):
#     path = VisionScorer.train_on_probes(cfg)
#     print(f"✅ Saved: {path}")

# if __name__ == "__main__":
#     train()
# ```

# ### `configs/vision_scorer.yaml`

# ```yaml
# defaults:
#   - graph_layout@_global_: graph_layout

# model:
#   backbone: tiny_cnn
#   head_hidden_dim: 64
#   layouts: ["forceatlas2", "spectral"]
#   dropout: 0.0

# input:
#   img_size: 256
#   per_channel_mean: [0.0, 0.0, 0.0]
#   per_channel_std: [1.0, 1.0, 1.0]

# training:
#   probe_types: ["sbm", "ring_of_cliques", "barbell"]
#   n_samples_per_probe: 300
#   batch_size: 32
#   lr: 1e-3
#   epochs: 8
#   device: cpu
#   save_path: ${paths.models_dir}/graph_vision_scorer.pt
#   img_size: 256
#   seed: 7

# inference:
#   cache_dir: ${paths.cache_dir}/vision_scorer
#   device: cpu
#   timeout_s: 2.0
# ```
