# stephanie/components/gap/scm/head.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import math

SCM_DIMS = ["reasoning","knowledge","clarity","faithfulness","coverage"]
SCM_COLUMNS = [
    "scm.reasoning.score01","scm.knowledge.score01","scm.clarity.score01",
    "scm.faithfulness.score01","scm.coverage.score01",
    "scm.aggregate01","scm.uncertainty01","scm.ood_hat01",
    "scm.consistency01","scm.length_norm01","scm.temp01","scm.agree_hat01"
]

def _to01(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().mean().item()
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0

class SimpleProjector(nn.Module):
    """Light MLP to compress pooled features into a shared latent."""
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class UniversalSCMHead(nn.Module):
    """
    A small learned projector + fixed postprocessing to produce SCM metrics.
    It can run 'cold' (random init) with hand-crafted formulas, and later be
    fine-tuned on self-supervised/relative targets.
    """
    def __init__(self, in_dim: int = 768, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.projector = SimpleProjector(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
        # optional tiny heads for OOD/uncertainty if you decide to learn them
        self.ood_head = nn.Linear(out_dim, 1)
        self.cons_head = nn.Linear(out_dim, 1)

    def forward(self, tapped: Dict[str, Any]) -> Dict[str, float]:
        # 1) Pool inputs into a fixed vector
        feats: List[torch.Tensor] = []
        # latent traces
        if "latent" in tapped and tapped["latent"] is not None:
            lat = tapped["latent"]  # [D] or [L, D]
            if isinstance(lat, torch.Tensor):
                if lat.ndim == 2: lat = lat.mean(dim=0)
                feats.append(lat.float())
        # logits entropy
        if "final_logits" in tapped and tapped["final_logits"] is not None:
            logits = tapped["final_logits"]
            if isinstance(logits, torch.Tensor):
                probs = logits.log_softmax(-1).exp()
                ent = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)  # [T]
                feats.append(ent.mean().unsqueeze(0))
        # attention summaries if provided
        if "attn_stats" in tapped:
            attn = tapped["attn_stats"]
            # e.g., entropy of attention, focus ratio, etc., precomputed as floats
            for v in attn.values():
                if isinstance(v, torch.Tensor): feats.append(v.flatten().float().mean().unsqueeze(0))
                elif isinstance(v, (int, float)): feats.append(torch.tensor([float(v)]))

        if feats:
            pooled = torch.cat([f if f.ndim == 1 else f.view(-1) for f in feats], dim=0)
        else:
            pooled = torch.zeros(32)

        # Pad/crop to fixed in_dim
        in_dim = self.projector.net[0].in_features
        if pooled.numel() < in_dim:
            pooled = torch.nn.functional.pad(pooled, (0, in_dim - pooled.numel()))
        elif pooled.numel() > in_dim:
            pooled = pooled[:in_dim]

        z = self.projector(pooled)  # [out_dim]

        # 2) Derive intrinsic signals (uncertainty, consistency, length_norm, temp, agree_hat)
        # Prefer tapped diagnostics if present; otherwise derive fallback signals
        # Uncertainty: prefer token entropy or step variance
        unc = None
        if "token_entropies" in tapped and tapped["token_entropies"] is not None:
            te = tapped["token_entropies"]
            if isinstance(te, torch.Tensor): unc = te.mean().item()
            elif isinstance(te, (list, tuple)): unc = float(sum(te)/max(1,len(te)))
        elif "steps" in tapped and tapped["steps"]:
            # variance of step scores if you stash them
            vals = []
            for s in tapped["steps"]:
                v = s.get("score_raw", None)
                if v is not None: vals.append(float(v))
            if len(vals) >= 2:
                m = sum(vals)/len(vals)
                unc = sum((x-m)**2 for x in vals)/len(vals)
        if unc is None: unc = 0.0
        unc01 = max(0.0, min(1.0, unc / 5.0))  # crude normalization

        # Consistency: inverse of uncertainty or step agreement if provided
        cons01 = 1.0 - unc01
        if "consistency_hat" in tapped:
            cons01 = max(0.0, min(1.0, float(tapped["consistency_hat"])))

        # OOD: learnable probe + optional heuristic (distance to running mean)
        ood01 = torch.sigmoid(self.ood_head(z)).item()
        if "ood_hat" in tapped:
            ood01 = max(0.0, min(1.0, float(tapped["ood_hat"])))

        # Length/temperature/agreement
        len01 = max(0.0, min(1.0, float(tapped.get("len_effect", 0.0))))
        temp01 = max(0.0, min(1.0, float(tapped.get("temp", 0.0))))
        agree01 = max(0.0, min(1.0, float(tapped.get("agree_hat", 0.5))))

        # 3) Produce 5 dimension scores from intrinsic signals (no rubric scores)
        # Simple transparent mapping; you can replace with a tiny linear layer per dim later.
        dim_scores = {}
        for d in SCM_DIMS:
            score = 0.45*(1.0 - unc01) + 0.35*cons01 + 0.20*(1.0 - ood01)
            dim_scores[d] = float(max(0.0, min(1.0, score)))

        aggregate = float(sum(dim_scores.values()) / len(SCM_DIMS))

        return {
            "scm.reasoning.score01":   dim_scores["reasoning"],
            "scm.knowledge.score01":   dim_scores["knowledge"],
            "scm.clarity.score01":     dim_scores["clarity"],
            "scm.faithfulness.score01":dim_scores["faithfulness"],
            "scm.coverage.score01":    dim_scores["coverage"],
            "scm.aggregate01":         aggregate,
            "scm.uncertainty01":       float(unc01),
            "scm.ood_hat01":           float(ood01),
            "scm.consistency01":       float(cons01),
            "scm.length_norm01":       float(len01),
            "scm.temp01":              float(temp01),
            "scm.agree_hat01":         float(agree01),
        }
