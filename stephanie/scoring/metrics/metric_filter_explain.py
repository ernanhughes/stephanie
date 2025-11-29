# stephanie/scoring/metrics/metric_filter_explain.py
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None  # CSV still works without pandas

@dataclass
class ColumnDiag:
    name: str
    kept: bool
    reason: str                      # e.g., "low-variance", "non-finite", "duplicate(~0.998→HRM.coverage.score)", "selected"
    variance: float
    min: float
    max: float
    frac_zeros: float
    frac_ones: float
    auc: Optional[float] = None      # if labels exist
    mi: Optional[float] = None       # if labels exist

def _safe_stats(col: np.ndarray) -> Tuple[float, float, float, float, float]:
    x = np.asarray(col, dtype=float)
    x_ok = x[np.isfinite(x)]
    if x_ok.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    v = float(np.nanvar(x_ok))
    mn = float(np.nanmin(x_ok))
    mx = float(np.nanmax(x_ok))
    frac0 = float(np.mean(np.isclose(x_ok, 0.0))) if x_ok.size else float("nan")
    frac1 = float(np.mean(np.isclose(x_ok, 1.0))) if x_ok.size else float("nan")
    return v, mn, mx, frac0, frac1

def _label_scores(col: np.ndarray, y: Optional[Sequence[int]]) -> Tuple[Optional[float], Optional[float]]:
    try:
        if y is None:
            return None, None
        y = np.asarray(y, dtype=int)
        if y.size != col.shape[0] or y.min() == y.max():
            return None, None
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, col))
        except Exception:
            auc = None
        try:
            from sklearn.feature_selection import mutual_info_classif
            mi = float(mutual_info_classif(col.reshape(-1, 1), y, discrete_features=False, random_state=0)[0])
        except Exception:
            mi = None
        return auc, mi
    except Exception:
        return None, None

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def write_metric_filter_explain(
    *,
    run_dir: Path,
    names_union: List[str],
    rows: List[Dict[str, Any]],
    kept_names: List[str],
    dup_pairs: List[Tuple[str, str, float]],
    nonfinite_idx: Iterable[int],
    lowvar_idx: Iterable[int],
    labels: Optional[Sequence[int]],
    normalize_used: bool,
    rank_method: str,
    cfg_snapshot: Dict[str, Any] | None = None,
    md_filename: str = "metric_filter_explain.md",
    csv_filename: str = "metric_filter_explain.csv",
    always_include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Produces:
      - Markdown report (per-run)
      - CSV table of per-column diagnostics
    Returns a small summary dict you can stash into Memory.
    """
    # Build dense matrix on union for diagnostics
    name_to_pos = {n: i for i, n in enumerate(names_union)}
    X = np.zeros((len(rows), len(names_union)), dtype=float)
    for i, r in enumerate(rows):
        cols = r.get("metrics_columns") or []
        vals = r.get("metrics_values") or []
        mp = dict(zip(cols, vals))
        for n, j in name_to_pos.items():
            X[i, j] = float(mp.get(n, 0.0))

    kept_set = set(kept_names)
    nonfinite_set = set(int(i) for i in nonfinite_idx)
    lowvar_set = set(int(i) for i in lowvar_idx)
    dup_dropped_map = {}  # dropped -> (kept, sim)
    for kept, dropped, sim in dup_pairs:
        dup_dropped_map[dropped] = (kept, float(sim))

    # Diagnostics per column
    y = np.asarray(labels, dtype=int) if labels is not None else None
    diags: List[ColumnDiag] = []
    for j, n in enumerate(names_union):
        col = X[:, j]
        var, mn, mx, f0, f1 = _safe_stats(col)
        auc, mi = _label_scores(col, y)
        if j in nonfinite_set:
            reason = "non-finite"
        elif j in lowvar_set:
            reason = "low-variance"
        elif n in dup_dropped_map:
            k, sim = dup_dropped_map[n]
            reason = f"duplicate(~{sim:.3f}→{k})"
        elif n in kept_set:
            reason = "selected"
        else:
            reason = "dropped"
        diags.append(ColumnDiag(
            name=n, kept=(n in kept_set), reason=reason,
            variance=var, min=mn, max=mx, frac_zeros=f0, frac_ones=f1, auc=auc, mi=mi
        ))

    # CSV
    csv_path = run_dir / csv_filename
    _ensure_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "kept", "reason", "variance", "min", "max", "frac_zeros", "frac_ones", "auc", "mi"])
        for d in diags:
            w.writerow([d.name, int(d.kept), d.reason, d.variance, d.min, d.max, d.frac_zeros, d.frac_ones, d.auc if d.auc is not None else "", d.mi if d.mi is not None else ""])

    # Headline stats
    kept_hrm = sum(1 for d in diags if d.kept and d.name.lower().startswith("hrm"))
    kept_sicql = sum(1 for d in diags if d.kept and d.name.lower().startswith("sicql"))
    kept_tiny = sum(1 for d in diags if d.kept and d.name.lower().startswith("tiny"))

    # Markdown
    md_path = run_dir / md_filename
    _ensure_dir(md_path)
    total = len(names_union)
    kept_ct = len(kept_names)
    dropped_ct = total - kept_ct
    # top “all-ones/zeros” suspects
    most_ones = sorted(diags, key=lambda d: (math.isnan(d.frac_ones), -d.frac_ones))[:10]
    most_zeros = sorted(diags, key=lambda d: (math.isnan(d.frac_zeros), -d.frac_zeros))[:10]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Metric Filter Explain Report\n\n")
        f.write(f"- **Total columns seen (union)**: {total}\n")
        f.write(f"- **Kept**: {kept_ct}\n")
        f.write(f"- **Dropped**: {dropped_ct}\n")
        f.write(f"- **Normalize to [0,1]**: {normalize_used}\n")
        f.write(f"- **Ranking method**: {rank_method}\n")
        if always_include:
            f.write(f"- **Always-include columns**: {len(always_include)} (forced)\n")
        if cfg_snapshot:
            f.write(f"\n<details><summary>Config snapshot</summary>\n\n```json\n{json.dumps(cfg_snapshot, indent=2)}\n```\n</details>\n")

        f.write("\n## Kept by family\n")
        f.write(f"- HRM: {kept_hrm}\n")
        f.write(f"- SICQL: {kept_sicql}\n")
        f.write(f"- Tiny: {kept_tiny}\n")

        f.write("\n## Duplicate drops (sample)\n")
        for dropped, (k, sim) in list(dup_dropped_map.items())[:20]:
            f.write(f"- `{dropped}` → `{k}` (cos≈{sim:.3f})\n")

        f.write("\n## High-probably-saturated columns (top frac=1.0)\n")
        for d in most_ones:
            if not math.isnan(d.frac_ones) and d.frac_ones > 0.9:
                f.write(f"- `{d.name}` frac_ones={d.frac_ones:.3f} var={d.variance:.2e}\n")

        f.write("\n## High-probably-dead columns (top frac=0.0)\n")
        for d in most_zeros:
            if not math.isnan(d.frac_zeros) and d.frac_zeros > 0.9:
                f.write(f"- `{d.name}` frac_zeros={d.frac_zeros:.3f} var={d.variance:.2e}\n")

        f.write("\n---\n")
        f.write(f"Full per-column table: `{csv_path}`\n")

    return {
        "csv": str(csv_path),
        "md": str(md_path),
        "kept": kept_names,
        "totals": {"union": total, "kept": kept_ct, "dropped": dropped_ct},
    }
