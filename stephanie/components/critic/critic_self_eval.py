# stephanie/components/critic/critic_self_eval.py
from __future__ import annotations
import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

# ---------- small utilities ----------
def _hash(arr: List[str]) -> str:
    return hashlib.sha256("|".join(arr).encode()).hexdigest()[:12]

def ece_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask): 
            continue
        conf = np.mean(p[mask])
        acc  = np.mean(y_true[mask])
        ece += (np.sum(mask) / len(y_true)) * abs(acc - conf)
    return float(ece)

def lift_at_k(y_true: np.ndarray, p: np.ndarray, k: float = 0.1) -> float:
    n = len(y_true)
    top = int(max(1, round(n * k)))
    order = np.argsort(-p)[:top]
    prec_top = float(np.mean(y_true[order]))
    base_rate = float(np.mean(y_true))
    return prec_top / max(1e-9, base_rate)

def paired_bootstrap_metrics(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, B: int = 5000, seed: int = 42
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    diffs_auc, diffs_brier, diffs_lift = [], [], []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yb, a, b = y[idx], p_a[idx], p_b[idx]
        diffs_auc.append(roc_auc_score(yb, b) - roc_auc_score(yb, a))
        diffs_brier.append(brier_score_loss(yb, b) - brier_score_loss(yb, a))
        diffs_lift.append(lift_at_k(yb, b, 0.10) - lift_at_k(yb, a, 0.10))
    def ci(x): 
        lo, hi = np.percentile(x, [2.5, 97.5])
        return float(np.mean(x)), float(lo), float(hi), float(np.mean(np.array(x) > 0.0))
    d_auc_m, d_auc_lo, d_auc_hi, p_pos_auc = ci(diffs_auc)
    d_br_m,  d_br_lo,  d_br_hi,  p_pos_br  = ci(diffs_brier)
    d_lf_m,  d_lf_lo,  d_lf_hi,  p_pos_lf  = ci(diffs_lift)
    return {
        "delta_auc":   {"mean": d_auc_m, "lo": d_auc_lo, "hi": d_auc_hi, "p_gt_0": p_pos_auc},
        "delta_brier": {"mean": d_br_m,  "lo": d_br_lo,  "hi": d_br_hi,  "p_lt_0": 1.0 - p_pos_br}, # lower is better
        "delta_lift@10": {"mean": d_lf_m, "lo": d_lf_lo, "hi": d_lf_hi, "p_gt_0": p_pos_lf},
    }

@dataclass
class SelfEvalResult:
    # point metrics
    auc_curr: float
    auc_cand: float
    brier_curr: float
    brier_cand: float
    ece_curr: float
    ece_cand: float
    lift10_curr: float
    lift10_cand: float
    # deltas + uncertainty
    boots: Dict[str, Dict[str, float]]
    # single scalar competence score for the candidate
    competence: float
    feature_fingerprint: str

def competence_score(auc: float, ece: float, brier: float) -> float:
    # 0..1 score: discrimination ↑, calibration ↓
    # safe monotone transform with soft caps
    a = (auc - 0.5) / 0.5            # 0..1
    e = math.exp(-5.0 * min(ece, 0.2))
    b = math.exp(-5.0 * min(brier, 0.3))
    c = max(0.0, min(1.0, 0.6*a + 0.2*e + 0.2*b))
    return float(c)

def self_evaluate(y: np.ndarray, p_curr: np.ndarray, p_cand: np.ndarray,
                  feature_names: List[str]) -> SelfEvalResult:
    auc_c = roc_auc_score(y, p_curr); auc_n = roc_auc_score(y, p_cand)
    br_c  = brier_score_loss(y, p_curr); br_n = brier_score_loss(y, p_cand)
    ece_c = ece_binary(y, p_curr); ece_n = ece_binary(y, p_cand)
    l10_c = lift_at_k(y, p_curr, 0.10); l10_n = lift_at_k(y, p_cand, 0.10)
    boots = paired_bootstrap_metrics(y, p_curr, p_cand)
    comp  = competence_score(auc_n, ece_n, br_n)
    return SelfEvalResult(
        auc_curr=auc_c, auc_cand=auc_n, brier_curr=br_c, brier_cand=br_n,
        ece_curr=ece_c, ece_cand=ece_n, lift10_curr=l10_c, lift10_cand=l10_n,
        boots=boots, competence=comp, feature_fingerprint=_hash(feature_names),
    )

# simple EMA state
def update_competence_ema(state_path: Path, new_value: float, alpha: float = 0.2) -> float:
    prev = None
    if state_path.exists():
        try: prev = json.loads(state_path.read_text()).get("ema")
        except Exception: pass
    ema = new_value if prev is None else (alpha*new_value + (1-alpha)*prev)
    state_path.write_text(json.dumps({"ema": ema}, indent=2))
    return float(ema)
