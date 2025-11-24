# stephanie/components/critic/model/report.py
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stephanie.scoring.metrics.metric_importance import (
    compute_metric_importance, save_metric_importance_json)

log = logging.getLogger(__name__)

CORE_FEATURE_COUNT = 8 # Assuming this is consistent; could load from config if needed

def _ensure_dir(p: Path) -> None:
    """Ensure directory exists with proper error handling."""
    try:
        p.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log.error(f"Failed to create directory {p}: {e}")
        return False

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

def _make_cv(groups: np.ndarray | None, y: np.ndarray, n_splits: int = 5):
    if groups is not None and len(groups) == len(y) and len(set(groups)) >= n_splits:
        return GroupKFold(n_splits=n_splits), groups
    # fallback to stratified if no/insufficient groups
    return StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=0), None

def _eval_linear_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None):
    cv, grp = _make_cv(groups, y)
    # Use a pipeline to ensure scaler is fit inside each CV fold
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, penalty="l2", solver="liblinear", max_iter=500)
    )
    scores = cross_val_score(clf, X, y, cv=cv, groups=grp, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())

def run_ablations(X: np.ndarray, y: np.ndarray, names: list[str],
                  core_dim: int, groups: np.ndarray | None):
    core_idx = list(range(core_dim))
    dyn_idx  = list(range(core_dim, X.shape[1]))
    auc_core_mean, auc_core_std = _eval_linear_model(X[:, core_idx], y, groups)

    if dyn_idx:
        auc_dyn_mean, auc_dyn_std = _eval_linear_model(X[:, dyn_idx], y, groups)
    else:
        # Neutral baseline if no dynamic features
        auc_dyn_mean, auc_dyn_std = 0.5, 0.0

    auc_hyb_mean,  auc_hyb_std  = _eval_linear_model(X, y, groups)
    return {
        "core_only_auc_mean": auc_core_mean, "core_only_auc_std": auc_core_std,
        "dynamic_only_auc_mean": auc_dyn_mean, "dynamic_only_auc_std": auc_dyn_std,
        "hybrid_auc_mean": auc_hyb_mean, "hybrid_auc_std": auc_hyb_std,
    }

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

def generate_visicalc_hypothesis_report(
    evaluation_results: Dict[str, Any],
    out_dir: Path,
    core_dim: int = CORE_FEATURE_COUNT
) -> bool:
    """Generate a report that validates or invalidates the VisiCalc hypothesis"""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "visicalc_hypothesis_validation.md"
    log.info(f"📝 Generating VisiCalc hypothesis validation report: {report_path}")

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
            f.write(f"# Feature Selection Summary\n")
            f.write(f"**Dataset:** {dataset_info['samples']} samples, {dataset_info['features']} features\n")
            f.write(f"**Core features:** {dataset_info['core_features']} (always kept)\n")
            f.write(f"**Dynamic features selected:** {dataset_info['selected_dynamic']}/{dataset_info['total_dynamic']}\n")
            f.write("## Top Selected Features\n")
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
*Auto-generated by generate_visicalc_report.py*
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


# stephanie/reports/generate_visicalc_report.py (Corrected main function)
# ... (previous code remains the same until the main function) ...

def main(args):
    """Main function for dataset analysis and reporting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load the generated dataset
    dataset_path = Path("data/critic.npz")
    log.info(f"📈 Loading dataset for analysis: {dataset_path.absolute()}")
    try:
        with np.load(dataset_path, allow_pickle=True) as data:
            X = data['X']
            y = data['y']
            metric_names = data['metric_names'].tolist()
            groups = data.get('groups', None) # Load groups if saved
            if groups is not None:
                groups = np.array(groups) # Convert list back to numpy array
            else:
                log.info("ℹ️  Groups not found in dataset file, CV will use StratifiedKFold.")
    except FileNotFoundError:
        log.error(f"❌ Dataset file not found: {dataset_path}")
        raise
    except KeyError as e:
        log.error(f"❌ Missing key in dataset file {dataset_path}: {e}")
        raise

    # Check shapes after loading
    log.info(f"✅ Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")
    if groups is not None:
        log.info(f"✅ Loaded groups: groups.shape={groups.shape}")
    else:
        log.info("ℹ️  No groups loaded, using default CV.")

    # Ensure names 1:1 with columns
    assert X.shape[1] == len(metric_names), \
        f"num_metrics {X.shape[1]} != metric_names length {len(metric_names)}"

    # Core sanity checks (similar to generation script)
    core_dim = args.core_dim
    core_block = X[:, :core_dim]
    if np.isnan(core_block).any():
        raise ValueError("NaN in VisiCalc core features — check VisiCalc computation.")
    zero_var_core = int((core_block.std(axis=0) <= 1e-12).sum())
    if zero_var_core:
        log.warning("VisiCalc: %d core features are ~constant across samples", zero_var_core)

    # --- Reporting Logic ---
    root = Path("runs/visicalc") # Default output directory for reports

    # 1. Visualize features
    log.info("📊 Visualizing features to directory: {root}")
    _save_viz_plots(X, y, root)

    # 2. Generate comprehensive dataset report
    log.info("📊 Generating comprehensive dataset report...")
    generate_dataset_report(X, y, metric_names, root, save_plots=False) # Plots already saved above

    # 3. Comprehensive feature evaluation
    log.info("🔍 Starting comprehensive feature evaluation")
    evaluation_results = evaluate_all_features(
        X, y, metric_names,
        core_dim=core_dim,
        groups=groups # Pass groups if available
    )

    # 4. Validate VisiCalc hypothesis
    log.info("🔍 Validating VisiCalc hypothesis")
    hypothesis_validated = generate_visicalc_hypothesis_report(
        evaluation_results,
        root,
        core_dim=core_dim
    )

    # 5. Feature selection based on hypothesis validation
    log.info("🔍 Selecting features based on hypothesis validation")
    if hypothesis_validated:
        # Keep all core features + top dynamic metrics
        core_names = metric_names[:core_dim]
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

    # 6. Write selected feature artifacts
    dataset_info = {
        "samples": X.shape[0],
        "features": X.shape[1],
        "core_features": core_dim,
        "total_dynamic": len(metric_names) - core_dim,
        "selected_dynamic": len([n for n in selected_names if n not in metric_names[:core_dim]])
    }
    # Calculate importance rows for selected features (for artifacts)
    _, importance_rows = _select_features_via_importance_core_aware(
        X, y, metric_names, core_dim=core_dim, top_k_dynamic=args.top_k_dynamic, min_effect=args.min_effect
    )
    _write_selected_feature_artifacts(importance_rows, root, dataset_info)

    # 7. Write filtered NPZ (if requested)
    if args.write_filtered_npz:
        name_to_idx = {n: i for i, n in enumerate(metric_names)}
        sel_idx = [name_to_idx[n] for n in selected_names if n in name_to_idx]
        if len(sel_idx) == 0:
            log.error("❌ No features selected for filtered dataset")
        else:
            X_sel = X[:, sel_idx].astype(X.dtype, copy=False)
            filtered_path = root / f"visicalc_ab_dataset_core{core_dim}_dyn{len(sel_idx)-core_dim}.npz"
            np.savez_compressed(
                filtered_path,
                X=X_sel,
                y=y,
                metric_names=np.array(selected_names, dtype=object)
            )
            log.info(
                f"💾 Wrote filtered dataset with {len(sel_idx)} features "
                f"(core={core_dim}, dynamic={len(sel_idx)-core_dim}) → {filtered_path}"
            )

    # --- Write remaining artifacts ---
    out_dir = root / "feature_importance_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Importance (core + dynamic), all-features ranking, dynamic marginals ===
    rows_core, rows_dyn = importance_core_and_dynamic(
        X, y, metric_names, core_dim=core_dim,
        top_k_dynamic=min(50, max(1, X.shape[1]-core_dim)),  # just for report readability
        min_effect=0.0
    )
    rows_all = importance_all_features(X, y, metric_names)
    rows_dyn_marg = dynamic_marginals_with_core(X, y, metric_names, core_dim=core_dim)

    _write_csv(rows_core, out_dir / "core_features_importance.csv")
    _write_json(rows_core, out_dir / "core_features_importance.json")
    _write_csv(rows_dyn,  out_dir / "dynamic_features_importance.csv")
    _write_json(rows_dyn,  out_dir / "dynamic_features_importance.json")
    _write_csv(rows_all, out_dir / "all_features_importance.csv")
    _write_json(rows_all, out_dir / "all_features_importance.json")
    _write_csv(rows_dyn_marg, out_dir / "dynamic_marginals_with_core.csv")
    _write_json(rows_dyn_marg, out_dir / "dynamic_marginals_with_core.json")

    # === CV ablations (no trimming) ===
    cv_res = run_ablations(X, y, metric_names, core_dim=core_dim, groups=groups)
    (out_dir / "cv_ablations.json").write_text(json.dumps(cv_res, indent=2), encoding="utf-8")
    log.info("CV Ablations: %s", cv_res)

    log.info("🎉 Tiny Critic Dataset analysis and reporting completed successfully!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tiny Critic Dataset Analysis and Reporting")
    p.add_argument(
        "--core_dim",
        type=int,
        default=CORE_FEATURE_COUNT,
        help="Number of core VisiCalc features (default: 8)"
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