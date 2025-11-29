# stephanie/components/critic/reports/training_report.py

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix

log = logging.getLogger(__name__)


def generate_training_reports(
    model,
    feature_names: List[str],
    cv_summary: Dict[str, Any],
    holdout_summary: Dict[str, Any],
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    viz_dir: Path,
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Central entry point for Tiny Critic training reports.

    - Writes coefficient importance plot
    - Writes holdout confusion matrix plot
    - Writes holdout ROC curve
    - Writes a lightweight training_summary.json / .md

    This function is *best-effort* and should never raise.
    """
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        _plot_coefficients(
            model,
            feature_names,
            viz_dir / "coef_importance.png",
        )
    except Exception as e:
        log.warning("CriticTrainingReport: coef plot failed: %s", e)

    try:
        _plot_holdout_cm(
            model,
            X_holdout,
            y_holdout,
            viz_dir / "holdout_confusion.png",
        )
    except Exception as e:
        log.warning("CriticTrainingReport: confusion plot failed: %s", e)

    try:
        _plot_holdout_roc(
            model,
            X_holdout,
            y_holdout,
            viz_dir / "holdout_roc.png",
        )
    except Exception as e:
        log.warning("CriticTrainingReport: ROC plot failed: %s", e)

    try:
        _write_training_summary(
            viz_dir / "training_summary.json",
            viz_dir / "training_summary.md",
            feature_names,
            cv_summary,
            holdout_summary,
            extra_meta=extra_meta,
        )
    except Exception as e:
        log.warning("CriticTrainingReport: summary write failed: %s", e)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _plot_coefficients(
    model,
    feature_names: List[str],
    out_path: Path,
    top_k: int = 30,
) -> None:
    """
    Barplot of |coef| for top features.
    """
    # unwrap pipeline to the step with coef_
    lr = None
    steps = getattr(model, "steps", None)
    if steps is not None:
        for name, step in steps:
            if hasattr(step, "coef_"):
                lr = step
    if lr is None and hasattr(model, "coef_"):
        lr = model
    if lr is None:
        raise ValueError("CriticTrainingReport: no linear model with coef_ found")

    coefs = lr.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1][: min(top_k, len(coefs))]
    imp_names = [feature_names[i] for i in order]
    imp_vals = coefs[order]

    plt.figure(figsize=(8, max(4, len(order) * 0.25)))
    sns.barplot(x=np.abs(imp_vals), y=imp_names, orient="h")
    plt.title("TinyCritic | |coef| (top)")
    plt.xlabel("|coef|")
    plt.ylabel("feature")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    log.info("CriticTrainingReport: wrote coef plot → %s", out_path)


def _plot_holdout_cm(
    model,
    X: np.ndarray,
    y: np.ndarray,
    out_path: Path,
) -> None:
    """
    Confusion matrix on the holdout set.
    """
    if len(y) == 0 or len(np.unique(y)) < 2:
        log.info("CriticTrainingReport: skipping CM plot (degenerate labels)")
        return

    yhat = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(y, yhat)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("TinyCritic | Holdout Confusion")
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    log.info("CriticTrainingReport: wrote confusion matrix → %s", out_path)


def _plot_holdout_roc(
    model,
    X: np.ndarray,
    y: np.ndarray,
    out_path: Path,
) -> None:
    """
    ROC curve on the holdout set.
    """
    if len(y) == 0 or len(np.unique(y)) < 2:
        log.info("CriticTrainingReport: skipping ROC plot (degenerate labels)")
        return

    p = model.predict_proba(X)[:, 1]
    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y, p)
    plt.title("TinyCritic | Holdout ROC")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    log.info("CriticTrainingReport: wrote ROC plot → %s", out_path)


def _write_training_summary(
    json_path: Path,
    md_path: Path,
    feature_names: List[str],
    cv_summary: Dict[str, Any],
    holdout_summary: Dict[str, Any],
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Lightweight machine + human-readable summary of training.
    """
    summary: Dict[str, Any] = {
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "cv_summary": cv_summary or {},
        "holdout_summary": holdout_summary or {},
    }
    if extra_meta:
        summary["meta"] = dict(extra_meta)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Minimal markdown wrapper
    cv = summary["cv_summary"]
    ho = summary["holdout_summary"]
    md = []
    md.append("# TinyCritic — Training Summary\n")
    md.append(f"- **Features used:** {len(feature_names)}")
    if "auc_mean" in cv:
        md.append(f"- **CV AUC (mean ± std):** {cv.get('auc_mean'):.4f} ± {cv.get('auc_std', 0.0):.4f}")
    if "acc_mean" in cv:
        md.append(f"- **CV ACC (mean ± std):** {cv.get('acc_mean'):.4f} ± {cv.get('acc_std', 0.0):.4f}")
    if "auc" in ho:
        md.append(f"- **Holdout AUC:** {ho.get('auc'):.4f}")
    if "acc" in ho:
        md.append(f"- **Holdout ACC:** {ho.get('acc'):.4f}")
    if "n" in ho:
        md.append(f"- **Holdout size (n):** {ho.get('n')}")

    if extra_meta:
        md.append("\n## Meta\n")
        for k, v in extra_meta.items():
            md.append(f"- **{k}:** {v}")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("CriticTrainingReport: wrote training summary → %s, %s", json_path, md_path)
