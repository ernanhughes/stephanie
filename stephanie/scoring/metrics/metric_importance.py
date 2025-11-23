# stephanie/scoring/metrics/metric_importance.py
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class MetricImportance:
    name: str
    mean_target: float
    mean_baseline: float
    std_target: float
    std_baseline: float
    cohen_d: float
    abs_cohen_d: float
    ks_stat: float
    ks_pvalue: float
    auc: float  # probability metric_target > metric_baseline
    direction: int  # +1 if target > baseline, -1 if baseline > target, 0 if tie-ish

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _cohen_d(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return 0.0
    mx = float(x.mean())
    my = float(y.mean())
    vx = float(x.var(ddof=1))
    vy = float(y.var(ddof=1))
    # pooled std
    sp2 = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    sp = math.sqrt(max(sp2, eps))
    return (mx - my) / sp


def _ks_2sample(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Tiny self-contained 2-sample KS test (approx p-value).

    We could use scipy.stats.ks_2samp, but this keeps the dependency surface small.
    This is not meant to be super-precise; it's just a "is there a big distribution gap?" flag.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return 0.0, 1.0

    # Empirical CDFs
    data_all = np.concatenate([x, y])
    data_sorted = np.sort(data_all)
    nx = x.size
    ny = y.size

    # ranks → CDF
    cdf_x = np.searchsorted(np.sort(x), data_sorted, side="right") / nx
    cdf_y = np.searchsorted(np.sort(y), data_sorted, side="right") / ny

    diffs = np.abs(cdf_x - cdf_y)
    d = float(diffs.max())

    # Rough asymptotic p-value (Smirnov)
    en = math.sqrt(nx * ny / (nx + ny))
    try:
        p = 2.0 * math.exp(-2.0 * (d * en) ** 2)
    except OverflowError:
        p = 0.0
    p = max(0.0, min(1.0, p))
    return d, p


def _auc_from_scores(x: np.ndarray, y: np.ndarray) -> float:
    """
    AUC-style probability that a random target > random baseline.
    Uses Mann–Whitney U on 1D feature only.

    Returns:
        auc \in [0,1], where 0.5 ~ no separation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx = x.size
    ny = y.size
    if nx == 0 or ny == 0:
        return 0.5

    # rank all values
    values = np.concatenate([x, y])
    ranks = values.argsort().argsort().astype(np.float64) + 1.0  # 1-based ranks
    r_x = ranks[:nx].sum()

    # Mann–Whitney U for x
    u_x = r_x - nx * (nx + 1) / 2.0
    auc = u_x / (nx * ny)
    return float(max(0.0, min(1.0, auc)))


def compute_metric_importance(
    X_target: np.ndarray,
    X_baseline: np.ndarray,
    metric_names: Sequence[str],
    *,
    top_k: int | None = None,
    min_effect: float = 0.0,
) -> List[MetricImportance]:
    """
    GAP-style per-metric importance analysis.

    Args:
        X_target:   (N_t, D) matrix for "good" / targeted cohort
        X_baseline: (N_b, D) matrix for baseline / "bad" cohort
        metric_names: length-D list of names
        top_k:      optionally keep only top_k metrics by |Cohen's d|
        min_effect: optionally require |d| >= this threshold

    Returns:
        List[MetricImportance], sorted by abs_cohen_d descending.
    """
    Xt = np.asarray(X_target, dtype=np.float64)
    Xb = np.asarray(X_baseline, dtype=np.float64)
    metric_names = list(metric_names)

    if Xt.ndim != 2 or Xb.ndim != 2:
        raise ValueError(f"compute_metric_importance: expected 2D matrices, got {Xt.shape} and {Xb.shape}")
    if Xt.shape[1] != Xb.shape[1]:
        raise ValueError(f"compute_metric_importance: dim mismatch {Xt.shape[1]} vs {Xb.shape[1]}")
    if len(metric_names) != Xt.shape[1]:
        raise ValueError(
            f"metric_names length {len(metric_names)} != num_metrics {Xt.shape[1]}"
        )

    N_t, D = Xt.shape
    N_b, _ = Xb.shape
    if N_t == 0 or N_b == 0 or D == 0:
        return []

    out: List[MetricImportance] = []
    for j in range(D):
        name = metric_names[j]
        tcol = Xt[:, j]
        bcol = Xb[:, j]

        m_t = float(np.mean(tcol))
        m_b = float(np.mean(bcol))
        s_t = float(np.std(tcol))
        s_b = float(np.std(bcol))

        d = _cohen_d(tcol, bcol)
        ks_stat, ks_p = _ks_2sample(tcol, bcol)
        auc = _auc_from_scores(tcol, bcol)

        # direction: which way is "better"?
        if abs(d) < 1e-6:
            direction = 0
        else:
            direction = 1 if d > 0 else -1

        imp = MetricImportance(
            name=name,
            mean_target=m_t,
            mean_baseline=m_b,
            std_target=s_t,
            std_baseline=s_b,
            cohen_d=d,
            abs_cohen_d=abs(d),
            ks_stat=ks_stat,
            ks_pvalue=ks_p,
            auc=auc,
            direction=direction,
        )
        out.append(imp)

    # sort most discriminative first
    out.sort(key=lambda r: r.abs_cohen_d, reverse=True)

    # apply filters
    if min_effect > 0.0:
        out = [r for r in out if r.abs_cohen_d >= min_effect]
    if top_k is not None and top_k > 0:
        out = out[:top_k]

    return out


def save_metric_importance_json(
    importance: List[MetricImportance],
    path: Path | str,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [m.to_dict() for m in importance]
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return p
