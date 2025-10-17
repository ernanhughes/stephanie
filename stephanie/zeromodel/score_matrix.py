# stephanie/eval/score_matrix.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Stephanie types (import if available; otherwise keep this file self-contained)
try:
    from stephanie.constants import GOAL, GOAL_TEXT
except Exception:
    GOAL, GOAL_TEXT = "goal", "goal_text"

try:
    from stephanie.scoring.scorable import (Scorable, ScorableFactory,
                                            ScorableType)
except Exception:
    Scorable, ScorableFactory, ScorableType = None, None, None


# ---------------------------
# Helpers
# ---------------------------

@dataclass
class SimpleScorable:
    id: str
    text: str

def _make_scorable(text: str, idx: int):
    """
    Prefer ScorableFactory if available in your env, else fallback to a tiny shim.
    """
    if ScorableFactory and ScorableType:
        # minimal dict accepted by your factory
        return ScorableFactory.from_dict({"text": text, "id": f"resp_{idx}"}, ScorableType.DOCUMENT)
    return SimpleScorable(id=f"resp_{idx}", text=text)


def _robust_minmax(series: pd.Series, lo=10.0, hi=90.0) -> pd.Series:
    """
    Scale to [0,1] using robust percentiles; clamp outliers.
    Works even if the two models are on different raw scales.
    """
    s = series.astype(float)
    p_lo = np.nanpercentile(s, lo) if s.notna().any() else 0.0
    p_hi = np.nanpercentile(s, hi) if s.notna().any() else 1.0
    if abs(p_hi - p_lo) < 1e-12:
        p_hi = p_lo + 1.0
    s_clamped = s.clip(lower=p_lo, upper=p_hi)
    return (s_clamped - p_lo) / (p_hi - p_lo)


def _corr_safe(a: pd.Series, b: pd.Series) -> Tuple[float, float, float]:
    """
    Pearson / Spearman / Kendall τ with NaN safety.
    """
    a, b = a.astype(float), b.astype(float)
    if a.notna().sum() < 3 or b.notna().sum() < 3:
        return float("nan"), float("nan"), float("nan")
    pearson = float(a.corr(b, method="pearson"))
    spearman = float(a.corr(b, method="spearman"))
    kendall = float(a.corr(b, method="kendall"))
    return pearson, spearman, kendall


# ---------------------------
# Core
# ---------------------------

def build_score_matrix(
    *,
    responses: Iterable[str],
    goal_text: str,
    dimensions: List[str],
    scorers: Dict[str, object],   # e.g. {"hrm": hrm_scorer, "tiny": tiny_scorer}
    memory=None,
    logger=None,
    max_n: int = 500,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns:
      df: rows = response_id, columns = MultiIndex(model, dimension), values = scores
      metrics: dict with per-dimension agreement + distribution summaries
    """
    # 1) slice to 500 and make scorables
    responses = list(responses)[:max_n]
    scorables = [_make_scorable(r, i) for i, r in enumerate(responses)]

    # 2) shared context
    context = {GOAL: {GOAL_TEXT: goal_text, "id": "eval_goal"}}

    # 3) score loop
    cols = []
    data = []
    if show_progress:
        pbar = tqdm(total=len(scorables)*max(1, len(scorers)), desc="Scoring")
    else:
        pbar = None

    # Prepare a container: dict[row_id][(model, dim)] = score
    row_dicts: List[Dict[Tuple[str, str], float]] = [dict() for _ in scorables]

    for model_name, scorer in scorers.items():
        for i, sc in enumerate(scorables):
            try:
                bundle = scorer.score(context, sc, dimensions)
                # bundle.results: dict[dimension] -> ScoreResult
                for dim in dimensions:
                    sr = bundle.results.get(dim)
                    if sr is None:
                        continue
                    row_dicts[i][(model_name, dim)] = float(sr.score)
            except Exception as e:
                if logger:
                    logger.log("EvalScorerError", {"model": model_name, "dim_set": dimensions, "idx": i, "error": str(e)})
            if pbar: pbar.update(1)

    if pbar: pbar.close()

    # 4) assemble DataFrame
    all_cols = sorted({k for row in row_dicts for k in row.keys()})
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in all_cols} for row in row_dicts])
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["model", "dimension"])
    df.index = [f"resp_{i}" for i in range(len(df))]

    # 5) compute metrics
    metrics = {"per_model_distribution": {}, "agreement": {}, "inter_dim_corr": {}}

    # 5a) per-model distribution stats per dimension
    for model_name in sorted({m for m, _ in df.columns}):
        sub = df[model_name] if model_name in df.columns.get_level_values(0) else None
        if sub is None: continue
        mstats = {}
        for dim in dimensions:
            if dim not in sub.columns: continue
            s = sub[dim].astype(float)
            mstats[dim] = {
                "count": int(s.notna().sum()),
                "mean": float(np.nanmean(s)),
                "std": float(np.nanstd(s)),
                "min": float(np.nanmin(s)) if s.notna().any() else float("nan"),
                "p25": float(np.nanpercentile(s, 25)) if s.notna().any() else float("nan"),
                "p50": float(np.nanpercentile(s, 50)) if s.notna().any() else float("nan"),
                "p75": float(np.nanpercentile(s, 75)) if s.notna().any() else float("nan"),
                "max": float(np.nanmax(s)) if s.notna().any() else float("nan"),
            }
        metrics["per_model_distribution"][model_name] = mstats

    # 5b) HRM vs Tiny agreement (normalize each to 0..1 via robust minmax)
    if ("hrm" in df.columns.get_level_values(0)) and ("tiny" in df.columns.get_level_values(0)):
        agr = {}
        for dim in dimensions:
            if (("hrm", dim) not in df.columns) or (("tiny", dim) not in df.columns):
                continue
            a_raw = df[("hrm", dim)]
            b_raw = df[("tiny", dim)]
            a = _robust_minmax(a_raw)
            b = _robust_minmax(b_raw)

            pearson, spearman, kendall = _corr_safe(a, b)
            mae = float(np.nanmean(np.abs(a - b)))
            agr[dim] = {
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "kendall_tau": kendall,
                "mae_norm01": mae,
            }
        metrics["agreement"]["hrm_vs_tiny"] = agr

    # 5c) inter-dimension correlation matrices (per model)
    for model_name in sorted({m for m, _ in df.columns}):
        try:
            sub = df[model_name].astype(float)
            # Only keep columns with non-NaN variation
            keep = [c for c in sub.columns if sub[c].notna().sum() > 3 and sub[c].std(skipna=True) > 1e-9]
            if keep:
                metrics["inter_dim_corr"][model_name] = sub[keep].corr(method="spearman").to_dict()
        except Exception:
            pass

    return df, metrics


def save_score_matrix(df: pd.DataFrame, metrics: Dict, *, out_prefix: str):
    """
    Save both artifacts. MultiIndex is preserved in CSV; metrics in JSON.
    """
    df.to_csv(f"{out_prefix}_scores.csv", index=True)
    with open(f"{out_prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# ---------------------------
# Example usage (put this in your notebook/script)
# ---------------------------
"""
from stephanie.eval.score_matrix import build_score_matrix, save_score_matrix
from stephanie.scoring.hrm_scorer import HRMScorer
from stephanie.scoring.tiny_scorer import TinyScorer  # <-- whatever your tiny scorer class is

# 1) init scorers
hrm = HRMScorer(cfg_hrm, memory, container, logger)
tiny = TinyScorer(cfg_tiny, memory, container, logger)

# 2) prepare data
responses = load_my_500_responses()    # list[str], length >= 500
goal_text = "General helpfulness and correctness relative to the user query."
dimensions = ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]

# 3) run
df, metrics = build_score_matrix(
    responses=responses,
    goal_text=goal_text,
    dimensions=dimensions,
    scorers={"hrm": hrm, "tiny": tiny},
    memory=memory,
    logger=logger,
    max_n=500,
)

# 4) save
save_score_matrix(df, metrics, out_prefix="out/hrm_vs_tiny")

# 5) quick looks
print(df.head())
print(json.dumps(metrics["agreement"]["hrm_vs_tiny"], indent=2))
"""
