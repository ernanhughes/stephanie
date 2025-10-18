# stephanie/components/gap/shared_scm.py
from __future__ import annotations
<<<<<<< HEAD
from typing import Dict, Any, List
import numpy as np

# Fixed column order for guaranteed alignment
=======
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# ----------------------------
# Scale-aware normalization
# ----------------------------

@dataclass(frozen=True)
class Range:
    lo: float
    hi: float
    def clamp01(self, v: float) -> float:
        if not np.isfinite(v) or self.hi == self.lo:
            return 0.0
        x = (v - self.lo) / (self.hi - self.lo)
        return float(np.clip(x, 0.0, 1.0))

class ScoreNormalizer:
    """
    Normalizes raw scores into [0,1] using explicit per-source ranges.
    We distinguish between:
      - SCORE ranges (dimension scores like reasoning/clarity/etc.)
      - ATTR ranges (uncertainty/entropy/energy/ood/etc.) — default 0–1
    """
    def __init__(
        self,
        score_ranges: Optional[Dict[Tuple[str, Optional[str]], Range]] = None,
        attr_ranges: Optional[Dict[Tuple[str, Optional[str]], Range]] = None,
    ):
        # Defaults (override as needed)
        self.score_ranges = {
            ("LLM", None): Range(0.0, 100.0),  # your rubric heads
            ("HRM", None): Range(0.0, 10.0),   # HRM typically 0–10
            ("TINY", None): Range(0.0, 1.0),   # TinyRecursion heads 0–1
        }
        if score_ranges:
            self.score_ranges.update(score_ranges)

        # Attributes are almost always already 0–1
        self.attr_ranges = {
            ("LLM", None): Range(0.0, 1.0),
            ("HRM", None): Range(0.0, 1.0),
            ("TINY", None): Range(0.0, 1.0),
        }
        if attr_ranges:
            self.attr_ranges.update(attr_ranges)

    def _get_range(self, table: Dict[Tuple[str, Optional[str]], Range],
                   source: str, dimension: Optional[str]) -> Range:
        key = (source, dimension)
        if key in table:
            return table[key]
        key = (source, None)
        if key in table:
            return table[key]
        return Range(0.0, 1.0)

    def norm_score(self, value: Any, *, source: str, dimension: Optional[str]) -> float:
        try:
            v = float(value)
        except Exception:
            return 0.0
        rng = self._get_range(self.score_ranges, source, dimension)
        return rng.clamp01(v)

    def norm_attr(self, value: Any, *, source: str, name: Optional[str] = None) -> float:
        try:
            v = float(value)
        except Exception:
            return 0.0
        rng = self._get_range(self.attr_ranges, source, name)
        return rng.clamp01(v)

# Single module-level normalizer (configurable if needed)
NORMALIZER = ScoreNormalizer()

def _source_from_prefix(model_prefix: str) -> str:
    p = (model_prefix or "").strip().lower()
    if p.startswith("hrm"):
        return "HRM"
    if p.startswith("tiny"):
        return "TINY"
    if p.startswith("llm"):
        return "LLM"
    # Fallback: treat unknown as LLM-style unless configured otherwise
    return "LLM"

# ----------------------------
# SCM schema (unchanged)
# ----------------------------

>>>>>>> main
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

<<<<<<< HEAD
def _to01(x: Any) -> float:
    try:
        v = float(x)
        # crude: treat >1 as 0-100 scaling (many hrm.* come as 0-100)
        if v > 1.0:
            return float(np.clip(v / 100.0, 0.0, 1.0))
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return 0.0
=======
# ----------------------------
# Helpers (now scale-aware)
# ----------------------------
>>>>>>> main

def _mean01(vals: List[float], default: float = 0.0) -> float:
    arr = [v for v in vals if v is not None]
    if not arr:
        return default
    return float(np.clip(float(np.mean(arr)), 0.0, 1.0))

<<<<<<< HEAD
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
=======
def _fetch_any_score(vec: Dict[str, Any], keys: List[str], *, source: str,
                     dimension: Optional[str], default: float = 0.0) -> float:
    for k in keys:
        if k in vec:
            return NORMALIZER.norm_score(vec[k], source=source, dimension=dimension)
    return default

def _fetch_any_attr(vec: Dict[str, Any], keys: List[str], *, source: str,
                    name: Optional[str] = None, default: float = 0.0) -> float:
    for k in keys:
        if k in vec:
            return NORMALIZER.norm_attr(vec[k], source=source, name=name)
    return default

def _dim_score(vec: Dict[str, Any], model_prefix: str, dim: str) -> float:
    src = _source_from_prefix(model_prefix)
    # try common forms, then fallback to 0
    return _fetch_any_score(vec, [
        f"{model_prefix}.{dim}.score",
        f"{model_prefix}.{dim}.aggregate",
        f"{model_prefix}.{dim}",
    ], source=src, dimension=dim, default=0.0)

def _uncertainty(vec: Dict[str, Any], model_prefix: str) -> float:
    src = _source_from_prefix(model_prefix)
    # prefer dimension-specific signals if any; otherwise fall back to global variants
    cands = []
    for d in DIMENSIONS:
        for name, key in (
            ("energy",     f"{model_prefix}.{d}.attr.energy"),
            ("entropy",    f"{model_prefix}.{d}.attr.entropy"),
            ("uncertainty",f"{model_prefix}.{d}.attr.uncertainty"),
        ):
            if key in vec:
                cands.append(NORMALIZER.norm_attr(vec[key], source=src, name=name))
    if cands:
        return _mean01(cands, 0.5)
    # global-ish fallbacks
    return _fetch_any_attr(vec, [
        f"{model_prefix}.attr.energy",
        f"{model_prefix}.attr.entropy",
        f"{model_prefix}.uncertainty",
    ], source=src, name="uncertainty", default=0.5)

def _ood(vec: Dict[str, Any], model_prefix: str) -> float:
    src = _source_from_prefix(model_prefix)
>>>>>>> main
    cands = []
    for d in DIMENSIONS:
        k = f"{model_prefix}.{d}.attr.ood_hat"
        if k in vec:
<<<<<<< HEAD
            cands.append(_to01(vec[k]))
    if cands:
        return _mean01(cands, 0.0)
    return _fetch_any(vec, [
        f"{model_prefix}.attr.ood_hat",
        f"{model_prefix}.ood_hat",
    ], default=0.0)

def _consistency(vec: Dict[str, float], model_prefix: str) -> float:
=======
            cands.append(NORMALIZER.norm_attr(vec[k], source=src, name="ood_hat"))
    if cands:
        return _mean01(cands, 0.0)
    return _fetch_any_attr(vec, [
        f"{model_prefix}.attr.ood_hat",
        f"{model_prefix}.ood_hat",
    ], source=src, name="ood_hat", default=0.0)

def _consistency(vec: Dict[str, Any], model_prefix: str) -> float:
    src = _source_from_prefix(model_prefix)
>>>>>>> main
    cands = []
    for d in DIMENSIONS:
        k = f"{model_prefix}.{d}.attr.consistency_hat"
        if k in vec:
<<<<<<< HEAD
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
=======
            cands.append(NORMALIZER.norm_attr(vec[k], source=src, name="consistency_hat"))
    if cands:
        return _mean01(cands, 0.5)
    return _fetch_any_attr(vec, [
        f"{model_prefix}.attr.consistency_hat",
        f"{model_prefix}.consistency",
    ], source=src, name="consistency_hat", default=0.5)

def _len_eff(vec: Dict[str, Any], model_prefix: str) -> float:
    src = _source_from_prefix(model_prefix)
    return _fetch_any_attr(vec, [
        f"{model_prefix}.attr.len_effect",
        f"{model_prefix}.len_effect",
    ], source=src, name="len_effect", default=0.0)

def _temp(vec: Dict[str, Any], model_prefix: str) -> float:
    src = _source_from_prefix(model_prefix)
    # Already named "*01" in many pipelines; still treat as attr (0–1)
    return _fetch_any_attr(vec, [
        f"{model_prefix}.attr.temp01",
        f"{model_prefix}.temp01",
    ], source=src, name="temp01", default=0.0)

def _agree(vec: Dict[str, Any], model_prefix: str) -> float:
    src = _source_from_prefix(model_prefix)
    # accept both agree and disagree_hat
    if f"{model_prefix}.attr.agree01" in vec:
        return NORMALIZER.norm_attr(vec[f"{model_prefix}.attr.agree01"], source=src, name="agree01")
    if f"{model_prefix}.attr.disagree_hat" in vec:
        d = NORMALIZER.norm_attr(vec[f"{model_prefix}.attr.disagree_hat"], source=src, name="disagree_hat")
        return float(np.clip(1.0 - d, 0.0, 1.0))
    return 0.5

def _aggregate(vec: Dict[str, Any], model_prefix: str, dim_scores: Dict[str, float]) -> float:
    src = _source_from_prefix(model_prefix)
    agg_raw = None
    for k in (f"{model_prefix}.aggregate", f"{model_prefix}.score"):
        if k in vec:
            agg_raw = vec[k]; break
    if agg_raw is not None:
        # treat aggregate like a score (source-specific scale)
        return NORMALIZER.norm_score(agg_raw, source=src, dimension=None)
    return _mean01(list(dim_scores.values()), 0.0)

# ----------------------------
# Public API
# ----------------------------

def scm_from_vector(vec_native: Dict[str, Any], *, model_prefix: str) -> Dict[str, float]:
>>>>>>> main
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
