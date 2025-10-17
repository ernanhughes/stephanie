# stephanie/components/gap/shared_scm.py
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np

# Fixed column order for guaranteed alignment
SCM_COLUMNS = [
    "scm.reasoning.score01",
    "scm.knowledge.score01",
    "scm.clarity.score01",
    "scm.faithfulness.score01",
    "scm.coverage.score01",
    "scm.aggregate01",
    "scm.uncertainty01",
    "scm.ood_hat01",
    "scm.consistency01",
    "scm.length_norm01",
    "scm.temp01",
    "scm.agree_hat01",
]

DIMENSIONS = ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]

def _to01(x: Any) -> float:
    try:
        v = float(x)
        # crude: treat >1 as 0-100 scaling (many hrm.* come as 0-100)
        if v > 1.0:
            return float(np.clip(v / 100.0, 0.0, 1.0))
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return 0.0

def _mean01(vals: List[float], default: float = 0.0) -> float:
    arr = [v for v in vals if v is not None]
    if not arr:
        return default
    return float(np.clip(float(np.mean(arr)), 0.0, 1.0))

def _fetch_any(vec: Dict[str, float], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in vec:
            return _to01(vec[k])
    return default

def _dim_score(vec: Dict[str, float], model_prefix: str, dim: str) -> float:
    # try common forms, then fallback to 0
    return _fetch_any(vec, [
        f"{model_prefix}.{dim}.score",
        f"{model_prefix}.{dim}.aggregate",
        f"{model_prefix}.{dim}",
    ], default=0.0)

def _uncertainty(vec: Dict[str, float], model_prefix: str) -> float:
    # prefer dimension-specific signals if any; otherwise fall back to global variants
    cands = []
    for d in DIMENSIONS:
        for k in (f"{model_prefix}.{d}.attr.energy",
                  f"{model_prefix}.{d}.attr.entropy",
                  f"{model_prefix}.{d}.attr.uncertainty"):
            if k in vec:
                cands.append(_to01(vec[k]))
    if cands:
        return _mean01(cands, 0.5)
    # global-ish fallbacks
    return _fetch_any(vec, [
        f"{model_prefix}.attr.energy",
        f"{model_prefix}.attr.entropy",
        f"{model_prefix}.uncertainty",
    ], default=0.5)

def _ood(vec: Dict[str, float], model_prefix: str) -> float:
    cands = []
    for d in DIMENSIONS:
        k = f"{model_prefix}.{d}.attr.ood_hat"
        if k in vec:
            cands.append(_to01(vec[k]))
    if cands:
        return _mean01(cands, 0.0)
    return _fetch_any(vec, [
        f"{model_prefix}.attr.ood_hat",
        f"{model_prefix}.ood_hat",
    ], default=0.0)

def _consistency(vec: Dict[str, float], model_prefix: str) -> float:
    cands = []
    for d in DIMENSIONS:
        k = f"{model_prefix}.{d}.attr.consistency_hat"
        if k in vec:
            cands.append(_to01(vec[k]))
    if cands:
        return _mean01(cands, 0.5)
    return _fetch_any(vec, [
        f"{model_prefix}.attr.consistency_hat",
        f"{model_prefix}.consistency",
    ], default=0.5)

def _len_eff(vec: Dict[str, float], model_prefix: str) -> float:
    return _fetch_any(vec, [
        f"{model_prefix}.attr.len_effect",
        f"{model_prefix}.len_effect",
    ], default=0.0)

def _temp(vec: Dict[str, float], model_prefix: str) -> float:
    return _fetch_any(vec, [
        f"{model_prefix}.attr.temp01",
        f"{model_prefix}.temp01",
    ], default=0.0)

def _agree(vec: Dict[str, float], model_prefix: str) -> float:
    # accept both agree and disagree_hat
    if f"{model_prefix}.attr.agree01" in vec:
        return _to01(vec[f"{model_prefix}.attr.agree01"])
    if f"{model_prefix}.attr.disagree_hat" in vec:
        return float(np.clip(1.0 - _to01(vec[f"{model_prefix}.attr.disagree_hat"]), 0.0, 1.0))
    return 0.5

def _aggregate(vec: Dict[str, float], model_prefix: str, dim_scores: Dict[str, float]) -> float:
    agg = _fetch_any(vec, [f"{model_prefix}.aggregate", f"{model_prefix}.score"], default=-1.0)
    if agg >= 0.0:
        return agg
    return _mean01(list(dim_scores.values()), 0.0)

def scm_from_vector(vec_native: Dict[str, float], *, model_prefix: str) -> Dict[str, float]:
    # Tier 1: per-dimension
    dim_scores = {d: _dim_score(vec_native, model_prefix, d) for d in DIMENSIONS}
    # Tier 2: diagnostics
    unc = _uncertainty(vec_native, model_prefix)
    ood = _ood(vec_native, model_prefix)
    cons = _consistency(vec_native, model_prefix)
    length = _len_eff(vec_native, model_prefix)
    temp = _temp(vec_native, model_prefix)
    agree = _agree(vec_native, model_prefix)
    # Tier 3: anchor
    agg = _aggregate(vec_native, model_prefix, dim_scores)

    scm = {
        "scm.reasoning.score01": dim_scores["reasoning"],
        "scm.knowledge.score01": dim_scores["knowledge"],
        "scm.clarity.score01": dim_scores["clarity"],
        "scm.faithfulness.score01": dim_scores["faithfulness"],
        "scm.coverage.score01": dim_scores["coverage"],
        "scm.aggregate01": agg,
        "scm.uncertainty01": unc,
        "scm.ood_hat01": ood,
        "scm.consistency01": cons,
        "scm.length_norm01": length,
        "scm.temp01": temp,
        "scm.agree_hat01": agree,
    }
    return scm

def scm_row(scm_dict: Dict[str, float]) -> List[float]:
    return [float(scm_dict.get(k, 0.0)) for k in SCM_COLUMNS]
