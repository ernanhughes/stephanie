# stephanie/dataloaders/tiny_critic_dataset.py
from __future__ import annotations


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score


import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import csv
import numpy as np

import logging
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from stephanie.scoring.metrics.visicalc_features import extract_tiny_features
from stephanie.scoring.metrics.dynamic_features import (
    load_core_metric_names,
    build_dynamic_feature_vector,
)
from stephanie.scoring.metrics.metric_importance import compute_metric_importance, save_metric_importance_json

log = logging.getLogger(__name__)

CORE_METRIC_PATH = Path("config/core_metrics.json")
CORE_FEATURE_NAMES = [
    "stability", "middle_dip", "std_dev", "sparsity",
    "entropy", "trend", "mid_bad_ratio", "frontier_util"
]
CORE_FEATURE_COUNT = len(CORE_FEATURE_NAMES)

ALIAS_MAP = {
    "Tiny.coverage.attr.raw01": "Tiny.coverage.attr.tiny.score01",
    "Tiny.coverage.attr.values[0]": "Tiny.coverage.attr.tiny.score01",
    "Tiny.coverage.attr.vector.tiny.score01": "Tiny.coverage.attr.tiny.score01",
    "Tiny.coverage.score": "Tiny.coverage.attr.tiny.score01",
    "Tiny.coverage.attr.values[1]": "Tiny.coverage.attr.tiny.score100",
    "Tiny.coverage.attr.vector.tiny.score100": "Tiny.coverage.attr.tiny.score100",
}

def canonicalize_metric_names(names: list[str]) -> list[str]:
    return [ALIAS_MAP.get(n, n) for n in names]

def importance_core_and_dynamic(X: np.ndarray, y: np.ndarray,
                                metric_names: list[str], core_dim: int,
                                top_k_dynamic: int, min_effect: float):
    assert X.shape[1] == len(metric_names), \
        f"num_metrics {X.shape[1]} != metric_names length {len(metric_names)}"
    Xt, Xb = X[y == 1], X[y == 0]
    core_names    = list(metric_names[:core_dim])
    dynamic_names = list(metric_names[core_dim:])

    # full stats for core (report all)
    imps_core = compute_metric_importance(
        Xt[:, :core_dim], Xb[:, :core_dim], core_names,
        top_k=len(core_names), min_effect=0.0
    )
    # stats for dynamic (top_k for readability; not trimming yet)
    imps_dyn  = compute_metric_importance(
        Xt[:, core_dim:], Xb[:, core_dim:], dynamic_names,
        top_k=top_k_dynamic, min_effect=min_effect
    )

    def rowize(m, is_core: bool):
        d = m._asdict() if hasattr(m, "_asdict") else m.__dict__.copy()
        d["is_core"] = is_core
        d["name"] = d.get("name") or d.get("feature")
        return d

    rows_core = [rowize(m, True)  for m in imps_core]
    rows_dyn  = [rowize(m, False) for m in imps_dyn]
    add_effective_auc(rows_core)
    add_effective_auc(rows_dyn)
    return rows_core, rows_dyn

def importance_all_features(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[dict]:
    imps = compute_metric_importance(X[y==1], X[y==0], names,
                                     top_k=len(names), min_effect=0.0)
    rows = []
    for m in imps:
        d = m._asdict() if hasattr(m, "_asdict") else m.__dict__.copy()
        d["is_core"] = bool(names.index(m.name) < 8)  # safe if core_dim=8; pass in if needed
        d["name"] = d.get("name") or d.get("feature")
        rows.append(d)
    add_effective_auc(rows)
    return rows

def dynamic_marginals_with_core(X: np.ndarray, y: np.ndarray,
                                names: list[str], core_dim: int) -> list[dict]:
    Xt, Xb = X[y == 1], X[y == 0]
    dyn_rows = []
    for i, dyn_name in enumerate(names[core_dim:], start=core_dim):
        idx = list(range(core_dim)) + [i]
        sub_names = [names[j] for j in idx]
        imps = compute_metric_importance(Xt[:, idx], Xb[:, idx], sub_names,
                                         top_k=1, min_effect=0.0)
        if not imps:
            continue
        m = imps[0]
        d = m._asdict() if hasattr(m, "_asdict") else m.__dict__.copy()
        d["is_core"] = False
        d["name"] = d.get("name") or d.get("feature")
        add_effective_auc([d])
        dyn_rows.append(d)
    return dyn_rows

def add_effective_auc(rows: list[dict]) -> None:
    for r in rows:
        auc = r.get("auc")
        direction = r.get("direction")
        if auc is None or direction is None:
            r["effective_auc"] = None
        else:
            try:
                r["effective_auc"] = float(auc) if int(direction) == 1 else (1.0 - float(auc))
            except Exception:
                r["effective_auc"] = None

def _ensure_dir(p: Path) -> None:
    """Ensure directory exists with proper error handling."""
    try:
        p.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log.error(f"Failed to create directory {p}: {e}")
        return False

def _load_metric_means_from_csv(run_dir: Path, label: str) -> dict:
    """
    Load metric means from CSV matrix file.
    
    Handles:
      - visicalc_targeted_matrix.csv (label="targeted")
      - visicalc_baseline_matrix.csv (label="baseline")
    """
    csv_name = f"visicalc_{label}_matrix.csv"
    csv_path = run_dir / csv_name
    if not csv_path.exists():
        return {}

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if not header or header[0] != "scorable_id":
                raise ValueError(f"{csv_name} missing 'scorable_id' header")

            metric_names = header[1:]
            sums = [0.0] * len(metric_names)
            cnt = 0

            for row in r:
                if not row:
                    continue
                values = row[1:]
                # coerce to float; blank → 0.0
                vals = [float(v) if v not in ("", None) else 0.0 for v in values]
                if len(vals) != len(metric_names):
                    raise ValueError(f"{csv_name}: row width mismatch at line {cnt+2}")
                for i, v in enumerate(vals):
                    sums[i] += v
                cnt += 1

            if cnt == 0:
                return {}

            means = [s / cnt for s in sums]
            return dict(zip(metric_names, means))
    except Exception as e:
        log.error(f"Failed to load metric means from {csv_path}: {e}")
        return {}

def _dataset_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Cohen's d with robust edge case handling."""
    n1, n2 = len(a), len(b)
    
    # Handle empty arrays
    if n1 == 0 or n2 == 0:
        return 0.0
    
    # Handle single-element arrays
    if n1 == 1 and n2 == 1:
        return float(a[0] - b[0])
    
    # Calculate means
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    
    # Handle zero variance cases
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    
    # If both have zero variance, return raw difference
    if var_a < 1e-10 and var_b < 1e-10:
        return mean_a - mean_b
    
    # Calculate pooled standard deviation
    pooled_var = ((n1 - 1) * var_a + (n2 - 1) * var_b) / max(n1 + n2 - 2, 1)
    pooled_std = np.sqrt(max(pooled_var, 1e-10))
    
    return (mean_a - mean_b) / pooled_std

def _summarize_features(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[dict]:
    """Create comprehensive feature statistics report."""
    rows = []
    xb, xt = X[y == 0], X[y == 1]
    
    for i, name in enumerate(names):
        col_b = xb[:, i] if xb.size else np.empty((0,), dtype=np.float32)
        col_t = xt[:, i] if xt.size else np.empty((0,), dtype=np.float32)
        
        mean_b = float(np.mean(col_b)) if col_b.size else 0.0
        mean_t = float(np.mean(col_t)) if col_t.size else 0.0
        std_b = float(np.std(col_b, ddof=1)) if col_b.size > 1 else 0.0
        std_t = float(np.std(col_t, ddof=1)) if col_t.size > 1 else 0.0
        nz_b = int(np.count_nonzero(col_b))
        nz_t = int(np.count_nonzero(col_t))
        delta = mean_t - mean_b
        
        rows.append({
            "feature": name,
            "mean_baseline": mean_b,
            "std_baseline": std_b,
            "nonzero_baseline": nz_b,
            "mean_targeted": mean_t,
            "std_targeted": std_t,
            "nonzero_targeted": nz_t,
            "delta_mean_t_minus_b": delta,
            "abs_delta": abs(delta),
            "cohens_d": _dataset_cohens_d(col_t, col_b),
        })
    
    return rows

def _write_feature_stats_csv(rows: list[dict], out_csv: Path) -> None:
    """Write feature statistics to CSV with proper error handling."""
    try:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = list(rows[0].keys()) if rows else [
            "feature", "mean_baseline", "std_baseline", "nonzero_baseline",
            "mean_targeted", "std_targeted", "nonzero_targeted",
            "delta_mean_t_minus_b", "abs_delta", "cohens_d"
        ]
        
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
            
        log.info(f"📊 Wrote feature statistics to {out_csv}")
    except Exception as e:
        log.error(f"Failed to write feature stats CSV: {e}")

def _select_features_via_importance_core_aware(
    X: np.ndarray, 
    y: np.ndarray, 
    metric_names: list[str],
    core_dim: int = CORE_FEATURE_COUNT,
    top_k_dynamic: int = 30,
    min_effect: float = 0.0
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Core-aware feature selection that:
      - Always preserves the first `core_dim` features (VisiCalc core)
      - Selects top_k_dynamic features from the remaining metrics
      - Labels features as core/dynamic for clear reporting
    
    Returns:
      (selected_names, importance_rows)
    """
    log.info(f"🔍 Starting core-aware feature selection (core_dim={core_dim}, top_k_dynamic={top_k_dynamic})")
    
    # Validate inputs
    if core_dim < 0 or core_dim > len(metric_names):
        log.error(f"Invalid core_dim={core_dim} for {len(metric_names)} metrics")
        core_dim = min(max(0, core_dim), len(metric_names))
    
    # Split data by class
    Xt = X[y == 1]
    Xb = X[y == 0]
    
    # Split metrics into core and dynamic
    core_names = metric_names[:core_dim]
    dynamic_names = metric_names[core_dim:]
    
    log.info(f"📌 Core features ({len(core_names)}): {core_names}")
    log.info(f"📌 Dynamic features ({len(dynamic_names)}): {dynamic_names[:5]}... (truncated)")
    
    # Compute importance on dynamic metrics only
    importance_rows = []
    selected_dynamic = []
    
    if len(dynamic_names) > 0:
        try:
            imps = compute_metric_importance(
                Xt[:, core_dim:], 
                Xb[:, core_dim:], 
                dynamic_names,
                top_k=top_k_dynamic, 
                min_effect=min_effect
            )
            
            # Convert to list of dicts for easier processing
            importance_rows = []
            for m in imps:
                row = {
                    "name": m.name,
                    "mean_target": m.mean_target,
                    "mean_baseline": m.mean_baseline,
                    "std_target": m.std_target,
                    "std_baseline": m.std_baseline,
                    "cohen_d": m.cohen_d,
                    "abs_cohen_d": m.abs_cohen_d,
                    "ks_stat": m.ks_stat,
                    "ks_pvalue": m.ks_pvalue,
                    "auc": m.auc,
                    "direction": m.direction,
                    "is_core": False
                }
                importance_rows.append(row)
            
            selected_dynamic = [m.name for m in imps]
            log.info(f"✅ Selected {len(selected_dynamic)}/{len(dynamic_names)} dynamic features")
            
        except Exception as e:
            log.error(f"⚠️  Failed to compute metric importance: {e}")
            # Fallback to keeping all dynamic features if importance fails
            selected_dynamic = dynamic_names
            log.warning("⚠️  Falling back to all dynamic features")
    
    # Combine core + selected dynamic (preserving order)
    selected_names = core_names + selected_dynamic
    
    rows_core, rows_dyn_full = importance_core_and_dynamic(
        X, y, metric_names, core_dim=core_dim,
        top_k_dynamic=top_k_dynamic, min_effect=min_effect
    )

    importance_rows = rows_core[:]  # start with core rows (real stats)
    importance_rows.extend(
        [r for r in rows_dyn_full if r["name"] in selected_dynamic]
    )
    return selected_names, importance_rows

def _write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

def _write_json(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

def evaluate_all_features(
    X: np.ndarray, 
    y: np.ndarray, 
    metric_names: list[str],
    core_dim: int = CORE_FEATURE_COUNT,
    groups: np.ndarray | None = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation that properly validates all feature types
    
    Returns:
      {
        "all_features": list of importance metrics for all features,
        "core_features": importance metrics for core features only,
        "dynamic_features": importance metrics for dynamic features only,
        "ablation_results": {
            "core_only": (auc_mean, auc_std),
            "dynamic_only": (auc_mean, auc_std),
            "hybrid": (auc_mean, auc_std)
        }
      }
    """
    log.info("🔍 Starting comprehensive feature evaluation")
    
    # 1. Evaluate ALL features together (core + dynamic)
    log.info("📊 Evaluating ALL features together")
    all_imps = compute_metric_importance(
        X[y == 1], 
        X[y == 0], 
        metric_names,
        top_k=len(metric_names),
        min_effect=0.0
    )
    
    # 2. Evaluate core features in isolation
    log.info("📊 Evaluating core features in isolation")
    core_imps = compute_metric_importance(
        X[y == 1, :core_dim], 
        X[y == 0, :core_dim],
        metric_names[:core_dim],
        top_k=core_dim,
        min_effect=0.0
    )
    
    # 3. Evaluate dynamic features in isolation
    log.info("📊 Evaluating dynamic features in isolation")
    dynamic_imps = compute_metric_importance(
        X[y == 1, core_dim:], 
        X[y == 0, core_dim:], 
        metric_names[core_dim:],
        top_k=len(metric_names) - core_dim,
        min_effect=0.0
    )
    
    # 4. Run ablation study with proper grouping
    log.info("📊 Running ablation study with proper grouping")
    ablation_results = run_ablations(X, y, metric_names, core_dim, groups)
    
    # 5. Analyze core feature contributions
    log.info("📊 Analyzing core feature contributions")
    core_contributions = []
    for i, name in enumerate(metric_names[:core_dim]):
        # How much does this core feature improve performance when added to dynamic features?
        X_subset = np.column_stack([
            X[:, core_dim:],  # dynamic features
            X[:, i]            # this core feature
        ])
        names_subset = metric_names[core_dim:] + [name]
        
        auc_mean, auc_std = _eval_linear_model(X_subset, y, groups)
        
        # Baseline is dynamic features only
        baseline_auc = ablation_results["dynamic_only_auc_std"]

        improvement = auc_mean - baseline_auc
        
        core_contributions.append({
            "name": name,
            "improvement": improvement,
            "auc_mean": auc_mean,
            "auc_std": auc_std
        })
    
    return {
        "all_features": all_imps,
        "core_features": core_imps,
        "dynamic_features": dynamic_imps,
        "ablation_results": ablation_results,
        "core_contributions": core_contributions
    }

def generate_visicalc_validation_report(out_dir: Path, core_rows: list[dict],
                                        all_rows: list[dict], dyn_marginals: list[dict],
                                        core_dim: int, cv_core_auc: float | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Rank maps
    rank_all = {r["name"]: (i+1) for i, r in enumerate(all_rows)}
    # core stats
    sig_core = [r for r in core_rows
                if r.get("cohen_d") is not None
                and abs(float(r["cohen_d"])) >= 0.3
                and r.get("ks_pvalue") is not None
                and float(r["ks_pvalue"]) < 0.05]
    core_avg_d = float(np.mean([abs(float(r["cohen_d"])) for r in core_rows
                                if r.get("cohen_d") is not None])) if core_rows else 0.0

    # decision rule (tune if desired)
    hypothesis_supported = (len(sig_core) >= 3 and core_avg_d >= 0.3)

    lines = []
    lines.append("# VisiCalc Hypothesis Validation Report\n")
    lines.append("## Core features (VisiCalc) — stats\n")
    lines.append("| Feature | |d| | AUC | effAUC | KS p | Rank(all) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in core_rows:
        nm = r["name"]
        d  = r.get("cohen_d"); auc = r.get("auc"); eff = r.get("effective_auc"); p = r.get("ks_pvalue")
        rank = rank_all.get(nm, "-")
        lines.append(f"| {nm} | {abs(float(d)) if d is not None else 'NA'} | "
                     f"{auc if auc is not None else 'NA'} | "
                     f"{eff if eff is not None else 'NA'} | "
                     f"{p if p is not None else 'NA'} | {rank} |")

    if cv_core_auc is not None and cv_core_auc >= 0.75:
        hypothesis_supported = True

    lines.append("\n## Model-level (grouped CV)\n")
    if cv_core_auc is not None:
        lines.append(f"- Core-only AUC (GroupKFold): **{cv_core_auc:.3f}**")

    lines.append("\n## Summary\n")
    lines.append(f"- Significant core (|d|≥0.3 & p<0.05): **{len(sig_core)} / {core_dim}**")
    lines.append(f"- Avg |d| (core): **{core_avg_d:.3f}**")
    lines.append("\n### Decision\n")
    if hypothesis_supported:
        lines.append("✅ **HYPOTHESIS SUPPORTED** — proceed with Hybrid (Core + Dynamic).")
    else:
        lines.append("❌ **HYPOTHESIS NOT SUPPORTED** — investigate VisiCalc implementation or alternative structure.")

    (out_dir / "visicalc_hypothesis_validation.md").write_text("\n".join(lines), encoding="utf-8")

def generate_visicalc_hypothesis_report(
    evaluation_results: Dict[str, Any],
    out_dir: Path,
    core_dim: int = CORE_FEATURE_COUNT
) -> bool:
    """Generate a report that validates or invalidates the VisiCalc hypothesis"""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "visicalc_hypothesis_validation.md"
    
    # Extract key results
    core_imps = evaluation_results["core_features"]
    ablation = evaluation_results["ablation_results"]
    core_contribs = evaluation_results["core_contributions"]
    
    # Calculate key metrics
    significant_core = sum(
        1 for m in core_imps 
        if abs(m.cohen_d) > 0.3 and m.ks_pvalue < 0.05
    )
    
    core_avg_d = np.mean([abs(m.cohen_d) for m in core_imps])
    core_auc_improvement = ablation["hybrid_auc_std"] - ablation["dynamic_only_auc_std"]
    
    # Determine hypothesis status
    hypothesis_validated = (
        significant_core >= 2 and 
        core_avg_d > 0.25 and 
        core_auc_improvement > 0.05
    )
    
    # Generate report
    report = f"""# VisiCalc Hypothesis Validation Report

## Core Question
Do structural reasoning patterns (VisiCalc features) contain discriminative signals for reasoning quality?

## Methodology
We conducted a comprehensive evaluation:
1. ALL features evaluated together (core + dynamic)
2. ONLY core VisiCalc features evaluated
3. ONLY dynamic metrics evaluated
4. Ablation study comparing core-only, dynamic-only, and hybrid models
5. Core feature contribution analysis

## Results

### 1. Core VisiCalc Feature Performance
| Feature | Cohen's d | AUC | KS p-value | Contribution |
|---------|-----------|-----|------------|--------------|
"""
    
    # Sort core contributions by improvement
    core_contribs_sorted = sorted(
        core_contribs, 
        key=lambda x: x["improvement"], 
        reverse=True
    )
    
    for contrib in core_contribs_sorted:
        name = contrib["name"]
        core_imp = next((m for m in core_imps if m.name == name), None)
        
        cohen_d = core_imp.cohen_d if core_imp else "N/A"
        auc = core_imp.auc if core_imp else "N/A"
        ks_pvalue = core_imp.ks_pvalue if core_imp else "N/A"
        improvement = contrib["improvement"]
        
        report += f"| {name} | {cohen_d:.3f} | {auc:.3f} | {ks_pvalue:.4e} | {improvement:.4f} |\n"
    
    report += f"\n**Summary**: {significant_core}/{core_dim} core features show statistically significant discrimination (|d| > 0.3, p < 0.05)"

    # 2. Model Comparison
    report += f"""
    
### 2. Model Comparison
| Model | AUC (mean) | AUC (std) | Improvement vs Dynamic |
|-------|------------|-----------|------------------------|
| Core Only | {ablation['core_only_auc_mean']:.4f} | {ablation['core_only_auc_std']:.4f} | N/A |
| Dynamic Only | {ablation['dynamic_only_auc_mean']:.4f} | {ablation['dynamic_only_auc_std']:.4f} | 0.0000 |
| Hybrid (Core + Dynamic) | {ablation['hybrid_auc_mean']:.4f} | {ablation['hybrid_auc_std']:.4f} | {core_auc_improvement:.4f} |
"""
    
    # 3. Hypothesis Validation
    report += "\n### 3. VisiCalc Hypothesis Validation\n"
    
    if hypothesis_validated:
        report += "✅ **HYPOTHESIS SUPPORTED**: Structural reasoning patterns contain meaningful discriminative signals\n"
        report += f"- {significant_core}/{core_dim} core features show strong discrimination\n"
        report += f"- Average |Cohen's d| across core features: {core_avg_d:.3f}\n"
        report += f"- Hybrid model improves AUC by {core_auc_improvement:.4f} over dynamic-only model\n"
    else:
        report += "❌ **HYPOTHESIS NOT SUPPORTED**: Structural reasoning patterns lack sufficient discriminative power\n"
        report += f"- Only {significant_core}/{core_dim} core features show strong discrimination\n"
        report += f"- Average |Cohen's d| across core features: {core_avg_d:.3f}\n"
        report += f"- Hybrid model improves AUC by only {core_auc_improvement:.4f} over dynamic-only model\n"
    
    # 4. Recommended Next Steps
    report += "\n### 4. Recommended Next Steps\n"
    
    if hypothesis_validated:
        report += "1. **Proceed with VisiCalc integration**: Core features contain meaningful signals\n"
        report += "2. **Focus on top 3 core features**: " + ", ".join([c["name"] for c in core_contribs_sorted[:3]]) + "\n"
        report += "3. **Build hybrid critic**: Combine top core features with top dynamic metrics\n"
    else:
        report += "1. **Investigate why core features underperform**:\n"
        report += "   - Check VisiCalc implementation for errors\n"
        report += "   - Verify core feature calculations\n"
        report += "   - Consider alternative structural representations\n"
        report += "2. **Re-evaluate hypothesis**: Structural patterns may not be as discriminative as expected\n"
    
    # Save report
    report_path.write_text(report, encoding="utf-8")
    log.info(f"✅ Wrote VisiCalc hypothesis validation report to {report_path}")
    
    return hypothesis_validated

def _make_cv(groups: np.ndarray | None, y: np.ndarray, n_splits: int = 5):
    if groups is not None and len(groups) == len(y) and len(set(groups)) >= n_splits:
        return GroupKFold(n_splits=n_splits), groups
    # fallback to stratified if no/insufficient groups
    return StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=0), None

def _eval_linear_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None):
    cv, grp = _make_cv(groups, y)
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(C=0.5, penalty="l2", solver="liblinear", max_iter=500)
    scores = cross_val_score(clf, Xs, y, cv=cv, groups=grp, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())

def run_ablations(X: np.ndarray, y: np.ndarray, names: list[str],
                  core_dim: int, groups: np.ndarray | None):
    core_idx = list(range(core_dim))
    dyn_idx  = list(range(core_dim, X.shape[1]))
    auc_core_mean, auc_core_std = _eval_linear_model(X[:, core_idx], y, groups)
    auc_dyn_mean,  auc_dyn_std  = _eval_linear_model(X[:, dyn_idx] if dyn_idx else X[:, :0], y, groups)
    auc_hyb_mean,  auc_hyb_std  = _eval_linear_model(X, y, groups)
    return {
        "core_only_auc_mean": auc_core_mean, "core_only_auc_std": auc_core_std,
        "dynamic_only_auc_mean": auc_dyn_mean, "dynamic_only_auc_std": auc_dyn_std,
        "hybrid_auc_mean": auc_hyb_mean, "hybrid_auc_std": auc_hyb_std,
    }


def _write_selected_feature_artifacts(
    importance_rows: list[dict], 
    out_dir: Path,
    dataset_info: Dict[str, Any]
) -> None:
    """Write comprehensive feature selection artifacts with clear core/dynamic labeling."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON (full detail with dataset context)
    json_path = out_dir / "selected_features.json"
    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump({
                "dataset_info": dataset_info,
                "importance_rows": importance_rows
            }, f, indent=2)
        log.info(f"✅ Wrote detailed feature importance to {json_path}")
    except Exception as e:
        log.error(f"❌ Failed to write JSON feature report: {e}")
    
    # 2. CSV (quick scan with core/dynamic labeling)
    csv_path = out_dir / "selected_features.csv"
    try:
        cols = [
            "name", "is_core", "mean_target", "mean_baseline", "std_target", "std_baseline",
            "cohen_d", "abs_cohen_d", "ks_stat", "ks_pvalue", "auc", "direction"
        ]
        
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in importance_rows:
                # Convert None to empty string for CSV readability
                safe_row = {k: ("" if v is None else v) for k, v in row.items()}
                w.writerow(safe_row)
        
        log.info(f"✅ Wrote CSV feature report to {csv_path}")
    except Exception as e:
        log.error(f"❌ Failed to write CSV feature report: {e}")
    
    # 3. Names only (handy for downstream)
    names_path = out_dir / "selected_feature_names.txt"
    try:
        with names_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(row["name"] for row in importance_rows))
        log.info(f"✅ Wrote feature names to {names_path}")
    except Exception as e:
        log.error(f"❌ Failed to write feature names: {e}")
    
    # 4. Summary report (human-readable)
    summary_path = out_dir / "feature_selection_summary.md"
    try:
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(f"# Feature Selection Summary\n\n")
            f.write(f"**Dataset:** {dataset_info['samples']} samples, {dataset_info['features']} features\n")
            f.write(f"**Core features:** {dataset_info['core_features']} (always kept)\n")
            f.write(f"**Dynamic features selected:** {dataset_info['selected_dynamic']}/{dataset_info['total_dynamic']}\n\n")
            
            f.write("## Top Selected Features\n\n")
            f.write("| Feature | Type | Cohen's d | AUC | Direction |\n")
            f.write("|---------|------|-----------|-----|-----------|\n")
            
            # Show top 10 dynamic features
            dynamic_rows = [r for r in importance_rows if not r["is_core"]][:10]
            for row in dynamic_rows:
                f.write(
                    f"| {row['name']} | Dynamic | {row['cohen_d']:.3f} | {row['auc']:.3f} | {row['direction']} |\n"
                )
            
            # Always show core features too
            core_rows = [r for r in importance_rows if r["is_core"]]
            for row in core_rows:
                cd  = row.get("cohen_d"); auc = row.get("auc"); dr = row.get("direction")
                cd_s  = f"{cd:.3f}" if cd is not None else "N/A"
                auc_s = f"{auc:.3f}" if auc is not None else "N/A"
                dr_s  = f"{dr}" if dr is not None else "N/A"
                f.write(f"| {row['name']} | Core | {cd_s} | {auc_s} | {dr_s} |\n")
        
        log.info(f"✅ Wrote feature selection summary to {summary_path}")
    except Exception as e:
        log.error(f"❌ Failed to write feature selection summary: {e}")

def _save_viz_plots(X: np.ndarray, y: np.ndarray, out_dir: Path, max_feats: int = 12) -> list[str]:
    """Save visualization plots with robust error handling."""
    saved = []
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize for visualization (drop ~constant cols)
    std = X.std(axis=0)
    keep = std > 1e-8
    Xs = X[:, keep] if keep.any() else X[:, :0]
    
    if Xs.shape[1] == 0:
        log.warning("⚠️  All features are ~constant; skipping visualization plots")
        return saved
    
    # Histogram grid (KDE guarded)
    try:
        k = min(max_feats, Xs.shape[1])
        ncols = min(4, k)
        nrows = int(np.ceil(k / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
        
        for i in range(k):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            ax.hist(Xs[y == 0, i], bins=20, alpha=0.6, label="baseline")
            ax.hist(Xs[y == 1, i], bins=20, alpha=0.6, label="targeted")
            ax.set_title(f"feat_{i}")
        
        for j in range(k, nrows*ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")
        
        fig.legend()
        fig.tight_layout()
        hp = out_dir / "feature_hists.png"
        fig.savefig(hp.as_posix(), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(hp))
        log.info(f"🖼️ Saved feature histograms to {hp}")
    except Exception as e:
        log.error(f"❌ Failed to generate feature histograms: {e}")
    
    # PCA scatter (optional)
    try:
        if Xs.shape[1] >= 2:
            Xp = PCA(n_components=2, random_state=0).fit_transform(Xs)
            fig2 = plt.figure(figsize=(5, 4))
            ax2 = fig2.add_subplot(111)
            ax2.scatter(Xp[y == 0, 0], Xp[y == 0, 1], alpha=0.7, label="baseline")
            ax2.scatter(Xp[y == 1, 0], Xp[y == 1, 1], alpha=0.7, label="targeted")
            ax2.legend()
            ax2.set_title("PCA(2) of sanitized features")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            fig2.tight_layout()
            pp = out_dir / "pca_scatter.png"
            fig2.savefig(pp.as_posix(), dpi=150, bbox_inches="tight")
            plt.close(fig2)
            saved.append(str(pp))
            log.info(f"🖼️ Saved PCA scatter plot to {pp}")
        else:
            log.info("ℹ️  PCA skipped: <2 usable features")
    except Exception as e:
        log.error(f"❌ Failed to generate PCA scatter plot: {e}")
    
    return saved

def _write_dataset_report_md(
    out_md: Path, 
    X: np.ndarray, 
    y: np.ndarray,
    metric_names: list[str], 
    rows: list[dict],
    figs: list[str],
    dataset_info: Dict[str, Any]
) -> None:
    """Write comprehensive dataset report in Markdown format."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    
    targeted = int((y == 1).sum())
    baseline = int((y == 0).sum())
    rank = int(np.linalg.matrix_rank(X))
    zero_var = int((X.std(axis=0) <= 1e-8).sum())
    
    # Rank top features
    top_delta = sorted(rows, key=lambda r: r["abs_delta"], reverse=True)[:25]
    top_effect = sorted(rows, key=lambda r: abs(r["cohens_d"]), reverse=True)[:25]
    
    def rows_to_md(rr: list[dict]) -> str:
        hdr = "| feature | mean_b | mean_t | Δ | d | std_b | std_t |\n|---|---:|---:|---:|---:|---:|---:|"
        lines = [hdr]
        for r in rr:
            lines.append(
                f"| {r['feature']} | {r['mean_baseline']:.4f} | {r['mean_targeted']:.4f} "
                f"| {r['delta_mean_t_minus_b']:.4f} | {r['cohens_d']:.3f} "
                f"| {r['std_baseline']:.4f} | {r['std_targeted']:.4f} |"
            )
        return "\n".join(lines)
    
    figs_md = ""
    if figs:
        figs_md = "\n### Figures saved\n" + "\n".join(f"- {Path(p).name}" for p in figs) + "\n"
    
    # Calculate imbalance ratio
    imbalance_ratio = max(targeted, baseline) / min(targeted, baseline) if min(targeted, baseline) > 0 else float('inf')
    imbalance_msg = "balanced" if imbalance_ratio <= 1.5 else f"imbalanced (ratio={imbalance_ratio:.2f})"
    
    # Calculate feature density
    feature_density = 100 * np.mean(X != 0)
    
    md = f"""# Tiny Critic — Combined Dataset Audit

**Samples:** {X.shape[0]}  **Features:** {X.shape[1]}  **Rank:** {rank}  **Zero-variance:** {zero_var}
**Class balance:** {imbalance_msg} — baseline: {baseline}  targeted: {targeted}  (ratio={imbalance_ratio:.2f})
**Feature density:** {feature_density:.1f}% non-zero values
**Metric columns:** {len(metric_names)}  
(See `feature_stats.csv` for complete per-feature stats.)

## Top by |Δ mean| (targeted - baseline)
{rows_to_md(top_delta)}

## Top by effect size (Cohen's d)
{rows_to_md(top_effect)}
{figs_md}
---
*Auto-generated by tiny_critic_dataset.py*
"""
    
    try:
        out_md.write_text(md, encoding="utf-8")
        log.info(f"📝 Wrote dataset report to {out_md}")
    except Exception as e:
        log.error(f"❌ Failed to write dataset report: {e}")

def generate_dataset_report(
    X: np.ndarray, 
    y: np.ndarray, 
    metric_names: list[str],
    out_dir: Path, 
    save_plots: bool = False,
    dataset_info: Dict[str, Any] = None
) -> None:
    """Generate comprehensive dataset report with statistics and visualizations."""
    if dataset_info is None:
        dataset_info = {
            "samples": X.shape[0],
            "features": X.shape[1],
            "core_features": CORE_FEATURE_COUNT,
            "total_dynamic": len(metric_names) - CORE_FEATURE_COUNT,
            "selected_dynamic": len(metric_names) - CORE_FEATURE_COUNT
        }
    
    rows = _summarize_features(X, y, metric_names)
    stats_csv = out_dir / "feature_stats.csv"
    _write_feature_stats_csv(rows, stats_csv)
    
    figs = _save_viz_plots(X, y, out_dir) if save_plots else []
    _write_dataset_report_md(
        out_dir / "tiny_critic_report.md", 
        X, y, metric_names, rows, figs, dataset_info
    )

def load_metrics_for_run(run_dir: Path, label: str) -> dict:
    """
    Load metrics from CSV matrix first, then fall back to JSON.
    
    Returns:
      {metric_name: mean_value} dictionary
    """
    # 1) CSV means (new path)
    means = _load_metric_means_from_csv(run_dir, label)
    if means:
        log.info(f"✅ Loaded {len(means)} metrics from CSV for {label} in {run_dir.name}")
        return means
    
    # 2) Fallback JSON (legacy path)
    metrics_path = run_dir / f"metrics_{label}.json"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            log.info(f"✅ Loaded JSON metrics from {metrics_path}")
            return data
        except Exception as e:
            log.error(f"❌ Failed to load metrics from {metrics_path}: {e}")
    else:
        log.debug(f"ℹ️ No metrics file for {label} at {metrics_path}, using defaults")
    
    return {}

def load_visicalc_report(path: Path) -> dict:
    """Load VisiCalc report with detailed error handling."""
    log.info(f"📂 Loading VisiCalc report: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate expected structure
        required_keys = ['frontier', 'global', 'regions']
        if not all(k in data for k in required_keys):
            log.warning(f"⚠️  VisiCalc report missing expected keys: {required_keys}")
        
        log.info(
            f"✅ Successfully loaded VisiCalc report: {path.name} "
            f"(keys: {list(data.keys()) if data else 'empty'})"
        )
        return data
    except Exception as e:
        log.error(f"❌ Failed to load VisiCalc report {path}: {e}")
        raise

def collect_visicalc_samples(visicalc_root: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Collect VisiCalc samples with proper core/dynamic feature handling.
    
    Returns:
      (X, y, metric_names) where:
        - X: [n_samples, n_features] feature matrix
        - y: [n_samples] binary labels (1=targeted, 0=baseline)
        - metric_names: list of feature names in order
    """
    visicalc_root = Path(visicalc_root)
    X: List[np.ndarray] = []
    y: List[int] = []
    
    log.info(f"🔍 Starting VisiCalc sample collection from: {visicalc_root.absolute()}")
    
    if not visicalc_root.exists():
        log.error(f"❌ VisiCalc root directory not found: {visicalc_root}")
        raise FileNotFoundError(f"VisiCalc root not found: {visicalc_root}")
    
    # Count directories for progress tracking
    all_dirs = [d for d in sorted(visicalc_root.iterdir()) if d.is_dir()]
    log.info(f"📁 Found {len(all_dirs)} potential run directories to process")
    
    processed_runs = 0
    successful_runs = 0
    skipped_runs = 0
    
    # Get dynamic metric names (core features are handled internally)
    dynamic_metric_names = load_core_metric_names(CORE_METRIC_PATH)
    log.info(f"📌 Loaded {len(dynamic_metric_names)} dynamic metric names from MARS summary")
    
    # Complete feature names (core + dynamic)
    metric_names = CORE_FEATURE_NAMES + dynamic_metric_names
    log.info(f"📌 Total feature names: {len(metric_names)} (8 core + {len(dynamic_metric_names)} dynamic)")
    
    # Each subdirectory under runs/visicalc is a run_id (e.g. 8967)
    groups: list[str] = []
    for run_dir in all_dirs:
        processed_runs += 1
        log.info(f"🔄 Processing run {processed_runs}/{len(all_dirs)}: {run_dir.name}")
        
        targeted_path = run_dir / "visicalc_targeted.json"
        baseline_path = run_dir / "visicalc_baseline.json"
        
        # Check if required files exist
        if not targeted_path.exists():
            log.warning(
                f"⚠️  Skipping {run_dir.name}: targeted report missing at {targeted_path}"
            )
            skipped_runs += 1
            continue
        
        if not baseline_path.exists():
            log.warning(
                f"⚠️  Skipping {run_dir.name}: baseline report missing at {baseline_path}"
            )
            skipped_runs += 1
            continue
        
        # Load metrics first (may be empty dicts)
        targeted_metrics = load_metrics_for_run(run_dir, label="targeted")
        baseline_metrics = load_metrics_for_run(run_dir, label="baseline")
        
        try:
            log.info(f"📊 Loading reports for {run_dir.name}...")
            targeted_report = load_visicalc_report(targeted_path)
            baseline_report = load_visicalc_report(baseline_path)
            
            # Extract combined features for targeted (label=1)
            # IMPORTANT: Only pass dynamic metrics to build_dynamic_feature_vector
            # (core features are added internally)
            log.info(
                f"🔧 Building combined features for targeted report {targeted_path}"
            )
            xt = build_dynamic_feature_vector(
                targeted_report, 
                targeted_metrics, 
                dynamic_metric_names
            )
            
            # Validate feature vector length
            if len(xt) != len(metric_names):
                log.error(
                    f"❌ Feature vector length mismatch: "
                    f"expected {len(metric_names)}, got {len(xt)}"
                )
                # Attempt to fix by padding or truncating
                if len(xt) < len(metric_names):
                    xt = np.pad(xt, (0, len(metric_names) - len(xt)))
                else:
                    xt = xt[:len(metric_names)]
            
            X.append(xt)
            y.append(1)
            groups.append(run_dir.name)
            log.info(
                f"✅ Targeted combined features: shape={xt.shape}, dtype={xt.dtype}"
            )
            
            # Extract combined features for baseline (label=0)
            log.info(
                f"🔧 Building combined features for baseline report {baseline_path}"
            )
            xb = build_dynamic_feature_vector(
                baseline_report, 
                baseline_metrics, 
                dynamic_metric_names
            )
            
            # Validate feature vector length
            if len(xb) != len(metric_names):
                log.error(
                    f"❌ Feature vector length mismatch: "
                    f"expected {len(metric_names)}, got {len(xb)}"
                )
                # Attempt to fix by padding or truncating
                if len(xb) < len(metric_names):
                    xb = np.pad(xb, (0, len(metric_names) - len(xb)))
                else:
                    xb = xb[:len(metric_names)]
            
            X.append(xb)
            y.append(0)
            groups.append(run_dir.name)
            log.info(
                f"✅ Baseline combined features: shape={xb.shape}, dtype={xb.dtype}"
            )
            
            successful_runs += 1
            log.info(
                f"🎉 Successfully processed run {run_dir.name} "
                f"(total samples: {len(X)}, successful runs: {successful_runs})"
            )
        
        except Exception as e:
            log.error(f"❌ Failed to process run {run_dir.name}: {e}")
            skipped_runs += 1
            continue
    
    # Final summary
    log.info(
        "📊 Collection complete! Processed: %d, Successful: %d, Skipped: %d",
        processed_runs,
        successful_runs,
        skipped_runs,
    )
    
    if not X:
        log.error(f"❌ No VisiCalc samples found under {visicalc_root}")
        raise RuntimeError(f"No VisiCalc samples found under {visicalc_root}")
    
    # Convert to numpy arrays
    log.info(f"🔄 Converting {len(X)} samples to numpy arrays...")
    X_arr = np.stack(X, axis=0)
    y_arr = np.asarray(y, dtype=np.int64)
    
    log.info(
        "✅ Dataset created: X.shape=%s, y.shape=%s, dtypes: X=%s, y=%s",
        X_arr.shape,
        y_arr.shape,
        X_arr.dtype,
        y_arr.dtype,
    )
    
    # Log some statistics
    log.info(
        "📈 Dataset statistics: Targeted samples (y=1): %d, Baseline samples (y=0): %d, "
        "Feature range: [%.3f, %.3f]",
        int(np.sum(y_arr == 1)),
        int(np.sum(y_arr == 0)),
        float(X_arr.min()),
        float(X_arr.max()),
    )
    
    return X_arr, y_arr, metric_names, groups

def save_dataset(
    X: np.ndarray, 
    y: np.ndarray, 
    metric_names: List[str], 
    out_path: str | Path
) -> None:
    """Save dataset with metric names included and comprehensive logging."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save with complete feature names
        np.savez_compressed(
            out_path,
            X=X,
            y=y,
            metric_names=np.array(metric_names, dtype=object)
        )
        
        # Log file info
        file_size = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0
        log.info(
            "✅ Dataset saved successfully! File: %s Size: %.2f MB X.shape: %s y.shape: %s",
            out_path,
            file_size,
            X.shape,
            y.shape,
        )
    except Exception as e:
        log.error(f"❌ Failed to save dataset: {e}")
        raise

def safe_histplot(ax, x, y, bins=20):
    """
    Safe histogram plot that handles edge cases for KDE.
    
    Only attempts KDE when:
      - At least 5 unique values
      - Standard deviation > 1e-8
      - At least 10 samples
    """
    # Quick heuristics
    n_unique = np.unique(x).size
    std = float(np.std(x))
    try_kde = (n_unique >= 5) and (std > 1e-8) and (len(x) >= 10)
    
    try:
        sns.histplot(x=x, hue=y, kde=try_kde, bins=bins, ax=ax)
    except Exception as e:
        log.warning("⚠️  KDE failed (%s). Falling back to kde=False.", e)
        sns.histplot(x=x, hue=y, kde=False, bins=bins, ax=ax)

def visualize_features(X: np.ndarray, y: np.ndarray, out_dir: Path, max_feats: int = 12) -> None:
    """
    Save histogram grid and PCA scatter to files in out_dir with robust error handling.
    
    Handles:
      - Rank deficiency
      - Zero-variance features
      - Visualization edge cases
    """
    log.info(f"📊 Visualizing features to directory: {out_dir}")
    _ensure_dir(out_dir)
    
    # Rank/variance logging
    try:
        rank = np.linalg.matrix_rank(X)
        zero_var = int((X.std(axis=0) <= 1e-8).sum())
        log.info(f"📐 Feature matrix: shape={X.shape}, rank={rank}, zero-variance={zero_var}")
    except Exception as e:
        log.warning(f"⚠️  Failed to compute matrix rank: {e}")
        rank = "N/A"
        zero_var = "N/A"
    
    # Sanitize for visualization only
    try:
        Xs, keep = sanitize_features_for_viz(X, min_std=1e-8)
        if Xs.shape[1] == 0:
            log.warning("⚠️  All features are ~constant; skipping plots.")
            return
    except Exception as e:
        log.error(f"❌ Failed to sanitize features for visualization: {e}")
        return
    
    # Histogram grid (KDE guarded)
    try:
        k = min(max_feats, Xs.shape[1])
        ncols = min(4, k)
        nrows = int(np.ceil(k / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
        
        for i in range(k):
            r, c = divmod(i, ncols)
            safe_histplot(axes[r][c], Xs[:, i], y, bins=20)
            axes[r][c].set_title(f"feat_{i}")
        
        for j in range(k, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")
        
        fig.tight_layout()
        hist_path = out_dir / "feature_hists.png"
        fig.savefig(hist_path.as_posix(), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"🖼️ Saved histograms to {hist_path}")
    except Exception as e:
        log.error(f"❌ Failed to generate feature histograms: {e}")
    
    # PCA scatter (optional)
    try:
        if Xs.shape[1] >= 2:
            Xp = PCA(n_components=2, svd_solver="auto", random_state=0).fit_transform(Xs)
            fig2 = plt.figure(figsize=(5, 4))
            ax = fig2.add_subplot(111)
            ax.scatter(Xp[y == 0, 0], Xp[y == 0, 1], alpha=0.7, label="baseline (y=0)")
            ax.scatter(Xp[y == 1, 0], Xp[y == 1, 1], alpha=0.7, label="targeted (y=1)")
            ax.legend()
            ax.set_title("PCA(2) of sanitized features")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            fig2.tight_layout()
            pca_path = out_dir / "pca_scatter.png"
            fig2.savefig(pca_path.as_posix(), dpi=150, bbox_inches="tight")
            plt.close(fig2)
            log.info(f"🖼️ Saved PCA to {pca_path}")
        else:
            log.info("ℹ️  PCA skipped: <2 usable features")
    except Exception as e:
        log.warning(f"⚠️  PCA viz skipped: {e}")

def sanitize_features_for_viz(X: np.ndarray, min_std: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove features with ~zero variance; return (X_sanitized, keep_mask).
    This does NOT touch training; it's only for visualization safety.
    """
    std = X.std(axis=0)
    keep = std > min_std
    Xs = X[:, keep] if keep.any() else X[:, :0]
    log.info(f"🧹 Viz sanitize: kept {int(keep.sum())}/{X.shape[1]} features (min_std={min_std:.1e})")
    return Xs, keep

def main(args):
    """Main function with comprehensive logging and error handling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    
    root = Path("runs/visicalc")
    out = Path("data/tiny_visicalc_critic.npz")
    
    log.info("🚀 Starting Tiny Critic Dataset creation...")
    log.info(f"Source: {root.absolute()}")
    log.info(f"Destination: {out.absolute()}")
    
    try:
        # 1. Collect dataset
        X, y, metric_names, groups = collect_visicalc_samples(root)
        
        # 1b. Canonicalize once; use everywhere below
        metric_names = canonicalize_metric_names(list(metric_names))
        
        # 2. Visualize features
        visualize_features(X, y, root)
        
        # 3. Save full dataset
        save_dataset(X, y, metric_names, out)
        
        # 4. Comprehensive feature evaluation
        log.info("🔍 Starting comprehensive feature evaluation")
        evaluation_results = evaluate_all_features(
            X, y, metric_names, 
            core_dim=args.core_dim,
            groups=groups
        )
        
        # 5. Validate VisiCalc hypothesis
        log.info("🔍 Validating VisiCalc hypothesis")
        hypothesis_validated = generate_visicalc_hypothesis_report(
            evaluation_results, 
            root,
            core_dim=args.core_dim
        )
        
        # 6. Feature selection based on hypothesis validation
        log.info("🔍 Selecting features based on hypothesis validation")
        if hypothesis_validated:
            # Keep all core features + top dynamic metrics
            core_names = metric_names[:args.core_dim]
            dynamic_imps = sorted(
                evaluation_results["dynamic_features"], 
                key=lambda m: abs(m.cohen_d), 
                reverse=True
            )
            selected_dynamic = [m.name for m in dynamic_imps[:args.top_k_dynamic]]
            selected_names = core_names + selected_dynamic
        else:
            # Fall back to dynamic metrics only
            all_imps = sorted(
                evaluation_results["all_features"], 
                key=lambda m: abs(m.cohen_d), 
                reverse=True
            )
            selected_names = [m.name for m in all_imps[:args.top_k_dynamic]]        

        # 7. Write filtered NPZ (if requested)
        if args.write_filtered_npz:
            name_to_idx = {n: i for i, n in enumerate(metric_names)}
            sel_idx = [name_to_idx[n] for n in selected_names if n in name_to_idx]
            
            if len(sel_idx) == 0:
                log.error("❌ No features selected for filtered dataset")
            else:
                X_sel = X[:, sel_idx].astype(X.dtype, copy=False)
                filtered_path = root / f"visicalc_ab_dataset_core{args.core_dim}_dyn{len(sel_idx)-args.core_dim}.npz"
                
                np.savez_compressed(
                    filtered_path,
                    X=X_sel, 
                    y=y, 
                    metric_names=np.array(selected_names, dtype=object)
                )
                
                log.info(
                    f"💾 Wrote filtered dataset with {len(sel_idx)} features "
                    f"(core={args.core_dim}, dynamic={len(sel_idx)-args.core_dim}) → {filtered_path}"
                )
        
        # Ensure names 1:1 with columns
        assert X.shape[1] == len(metric_names), \
            f"num_metrics {X.shape[1]} != metric_names length {len(metric_names)}"

        # Canonicalize names for diagnostics (no trimming; no column drops)
        metric_names = canonicalize_metric_names(list(metric_names))

        # Core sanity
        core_dim = args.core_dim if hasattr(args, "core_dim") else 8
        core_block = X[:, :core_dim]
        if np.isnan(core_block).any():
            raise ValueError("NaN in VisiCalc core features — check VisiCalc computation.")
        zero_var_core = int((core_block.std(axis=0) <= 1e-12).sum())
        if zero_var_core:
            log.warning("VisiCalc: %d core features are ~constant across samples", zero_var_core)

        # === Importance (core + dynamic), all-features ranking, dynamic marginals ===
        rows_core, rows_dyn = importance_core_and_dynamic(
            X, y, metric_names, core_dim=core_dim,
            top_k_dynamic=min(50, max(1, X.shape[1]-core_dim)),  # just for report readability
            min_effect=0.0
        )
        rows_all = importance_all_features(X, y, metric_names)
        rows_dyn_marg = dynamic_marginals_with_core(X, y, metric_names, core_dim=core_dim)

        out_dir = Path(root) / "feature_importance_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(rows_core, out_dir / "core_features_importance.csv")
        _write_json(rows_core, out_dir / "core_features_importance.json")
        _write_csv(rows_dyn,  out_dir / "dynamic_features_importance.csv")
        _write_json(rows_dyn, out_dir / "dynamic_features_importance.json")
        _write_csv(rows_all, out_dir / "all_features_importance.csv")
        _write_json(rows_all, out_dir / "all_features_importance.json")
        _write_csv(rows_dyn_marg, out_dir / "dynamic_marginals_with_core.csv")
        _write_json(rows_dyn_marg, out_dir / "dynamic_marginals_with_core.json")

        # === Validation report ===
        generate_visicalc_validation_report(out_dir, rows_core, rows_all, rows_dyn_marg, core_dim=core_dim)

        # === CV ablations (no trimming) ===
        cv_res = run_ablations(X, y, metric_names, core_dim=core_dim, groups=groups)
        (out_dir / "cv_ablations.json").write_text(json.dumps(cv_res, indent=2), encoding="utf-8")
        log.info("CV Ablations: %s", cv_res)
        generate_visicalc_validation_report(
            out_dir, rows_core, rows_all, rows_dyn_marg, core_dim=core_dim,
            cv_core_auc=cv_res["core_only_auc_mean"]
        )

        log.info("🎉 Tiny Critic Dataset creation completed successfully!")

    except Exception as e:
        log.error(f"💥 Failed to create Tiny Critic Dataset: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tiny Critic Dataset Builder")
    p.add_argument(
        "--core_dim", 
        type=int, 
        default=CORE_FEATURE_COUNT,
        help="Number of core VisiCalc features to always keep (default: 8)"
    )
    p.add_argument(
        "--top_k_dynamic", 
        type=int, 
        default=30,
        help="Number of top dynamic metrics to keep based on importance"
    )
    p.add_argument(
        "--min_effect", 
        type=float, 
        default=0.0,
        help="Minimum Cohen's d effect size to include dynamic metrics"
    )
    p.add_argument(
        "--write_filtered_npz", 
        action="store_true",
        help="Write a filtered NPZ with core + top_k_dynamic features"
    )
    
    main(p.parse_args())