# stephanie/components/critic/model/trainer.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     GroupShuffleSplit, StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------
log = logging.getLogger(__name__)

DATA_PATH   = Path("data/critic.npz")
MODEL_PATH  = Path("models/critic.joblib")
META_PATH   = Path("models/critic.meta.json")
VIZ_DIR     = Path("data/visualizations/tiny_critic")

# The first 8 columns in X are VisiCalc core (by dataset construction)
CORE_FEATURE_NAMES = [
    "stability",
    "middle_dip",
    "std_dev",
    "sparsity",
    "entropy",
    "trend",
    "mid_bad_ratio",
    "frontier_util",
]

# Note: 1 means “higher = better”, -1 means “higher = worse” (so we’ll flip)
CORE_FEATURE_DIRECTION: Dict[str, int] = {
    "stability":      1,
    "middle_dip":     1,
    "std_dev":       -1,  # strongest, but inverted
    "sparsity":       1,
    "entropy":        1,
    "trend":          1,
    "mid_bad_ratio": -1,
    "frontier_util":  1,
}


# -------------------------------------------------------------------
# IO
# -------------------------------------------------------------------
def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], Optional[np.ndarray]]:
    """
    Load NPZ dataset. Expects keys: X, y, metric_names, (optional) groups.
    """
    log.info("📂 Loading dataset from: %s", path.absolute())
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as data:
        X = data["X"]
        y = data["y"].astype(int)
        metric_names = data["metric_names"].tolist() if "metric_names" in data else [
            *CORE_FEATURE_NAMES, *[f"feat_{i}" for i in range(X.shape[1] - len(CORE_FEATURE_NAMES))]
        ] 
        groups = data.get("groups", None)
        if groups is not None:
            groups = np.array(groups)

    log.info("✅ Dataset: X=%s, y=%s, groups=%s", X.shape, y.shape, None if groups is None else groups.shape)
    uniques, counts = np.unique(y, return_counts=True)
    log.info("📊 Class distribution: %s", dict(zip(uniques.tolist(), counts.tolist())))
    return X, y, metric_names, groups


# -------------------------------------------------------------------
# Feature handling
# -------------------------------------------------------------------
def _indices_for(names: list[str], keep: list[str]) -> list[int]:
    idx = []
    pos = {n: i for i, n in enumerate(names)}
    for k in keep:
        if k in pos:
            idx.append(pos[k])
    return idx

def apply_directionality(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """
    Flip any feature with direction -1 so that 'higher = better'.
    We just multiply by -1 for those; StandardScaler will handle scaling later.
    """
    Xc = X.copy()
    for i, n in enumerate(feature_names):
        d = CORE_FEATURE_DIRECTION.get(n)
        if d == -1:
            Xc[:, i] = -Xc[:, i]
    return Xc


# -------------------------------------------------------------------
# Viz (core-only quick plots)
# -------------------------------------------------------------------
def safe_hist(ax, x0: np.ndarray, x1: np.ndarray, bins=20):
    ax.hist(x0, bins=bins, alpha=0.6, label="baseline")
    ax.hist(x1, bins=bins, alpha=0.6, label="targeted")
    ax.legend()

def visualize_core(X: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    n_core = min(8, X.shape[1])
    xb, xt = X[y == 0], X[y == 1]

    # Core dists
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.ravel()
    for i in range(n_core):
        ax = axes[i]
        safe_hist(ax, xb[:, i], xt[:, i], bins=20)
        ax.set_title(names[i])
    for j in range(n_core, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "core_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Simple correlation
    try:
        import pandas as pd
        core_cols = min(n_core, X.shape[1])
        df = pd.DataFrame(X[:, :core_cols], columns=names[:core_cols])
        corr = df.corr()
        plt.figure(figsize=(9, 7))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Core Feature Correlations")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "core_feature_correlations.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        log.warning("Heatmap failed: %s", e)


# -------------------------------------------------------------------
# CV helpers
# -------------------------------------------------------------------
def make_group_aware_split(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray], test_size=0.2, seed=42):
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(X, y, groups=groups))
        return train_idx, val_idx
    # fallback (no groups in dataset)
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int((1.0 - test_size) * n)
    return idx[:cut], idx[cut:]


def make_cv(groups: Optional[np.ndarray], y: np.ndarray, n_splits: int = 5):
    if groups is not None and len(np.unique(groups)) >= n_splits:
        return GroupKFold(n_splits=n_splits), groups
    return StratifiedKFold(n_splits=min(n_splits, len(np.unique(y))), shuffle=True, random_state=42), None


# -------------------------------------------------------------------
# Model training / tuning / evaluation
# -------------------------------------------------------------------
def tune_and_build_model(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray]):
    """
    Tiny grid over C with group-aware CV when groups are available.
    """
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000, solver="liblinear", class_weight="balanced", random_state=42
        ),
    )
    param_grid = {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}
    cv, grp = make_cv(groups, y, n_splits=5)
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=1)
    gs.fit(X, y, **({"groups": grp} if grp is not None else {}))
    log.info("🔧 Best C: %s (AUC=%.3f)", gs.best_params_["logisticregression__C"], gs.best_score_)
    return gs.best_estimator_


def eval_cv(model, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray]) -> dict:
    cv, grp = make_cv(groups, y, n_splits=5)
    aucs, accs = [], []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups=grp)):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:, 1]
        yhat = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[va], p))
        accs.append(accuracy_score(y[va], yhat))
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "n_splits": int(getattr(cv, "n_splits", 5))
    }


# -------------------------------------------------------------------
# Training entry
# -------------------------------------------------------------------
def train_tiny_critic(
    X: np.ndarray,
    y: np.ndarray,
    metric_names: list[str],
    groups: Optional[np.ndarray],
    core_only: bool = True,
    lock_features_path: Optional[Path] = None,
) -> tuple[Any, dict, dict]:
    """
    Trains the Tiny Critic with:
      - Directionality correction
      - Optional core-only mode
      - Optional locked feature set (names in a text file)
      - Group-aware split + CV
    Returns: (fitted_model, cv_summary, holdout_summary)
    """
    # 1) Directionality correction across ALL features where we know direction
    Xc = apply_directionality(X, metric_names)

    # 2) Optionally lock features by name (overrides core_only)
    if lock_features_path and lock_features_path.exists():
        locked = [ln.strip() for ln in lock_features_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        keep_idx = _indices_for(metric_names, locked)
        if not keep_idx:
            raise ValueError("Locked feature list did not match any metric_names.")
        Xc = Xc[:, keep_idx]
        names_used = [metric_names[i] for i in keep_idx]
        log.info("🔒 Using locked features (%d): %s", len(names_used), names_used[:12])
    elif core_only:
        keep_idx = list(range(min(8, Xc.shape[1])))
        Xc = Xc[:, keep_idx]
        names_used = metric_names[:len(keep_idx)]
        log.info("✨ Core-only training on %d features", len(names_used))
    else:
        names_used = list(metric_names)
        log.info("🧰 Training on all %d features", Xc.shape[1])

    # 3) Group-aware holdout split
    train_idx, val_idx = make_group_aware_split(Xc, y, groups, test_size=0.2, seed=42)
    Xtr, Xva = Xc[train_idx], Xc[val_idx]
    ytr, yva = y[train_idx], y[val_idx]
    grp_tr = groups[train_idx] if groups is not None else None

    # 4) Tune + fit
    model = tune_and_build_model(Xtr, ytr, grp_tr)

    # 5) CV summary on training subset (group aware)
    cv_summary = eval_cv(model, Xtr, ytr, grp_tr)

    # 6) Holdout summary (pure generalization)
    p_val = model.predict_proba(Xva)[:, 1]
    yhat_val = (p_val >= 0.5).astype(int)
    holdout = {
        "auc": float(roc_auc_score(yva, p_val)) if len(np.unique(yva)) > 1 else 0.0,
        "acc": float(accuracy_score(yva, yhat_val)),
        "n_val": int(len(yva))
    }

    # 7) Save artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "feature_names": names_used,
        "core_only": core_only,
        "locked_features_path": str(lock_features_path) if lock_features_path else None,
        "directionality": CORE_FEATURE_DIRECTION,
        "cv_summary": cv_summary,
        "holdout_summary": holdout,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("💾 Saved model → %s", MODEL_PATH)
    log.info("💾 Saved meta  → %s", META_PATH)

    # 8) Quick confusion matrix on holdout
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(yva, yhat_val)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Holdout Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "confusion_matrix_holdout.png", dpi=300, bbox_inches="tight")
    plt.close()

    return model, cv_summary, holdout


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Train Tiny Critic (group-aware, direction-corrected)")
    p.add_argument("--data", type=str, default=str(DATA_PATH))
    p.add_argument("--core-only", action="store_true", default=False,
                   help="Use only the 8 VisiCalc core features")
    p.add_argument("--lock-features", type=str, default="",
                   help="Path to a text file with feature names to lock (one per line)")
    args = p.parse_args()

    X, y, metric_names, groups = load_dataset(Path(args.data))
    visualize_core(X, y, metric_names)

    lock_path = Path(args.lock_features) if args.lock_features else None
    model, cv_summary, holdout = train_tiny_critic(
        X, y, metric_names, groups,
        core_only=bool(args.core_only),
        lock_features_path=lock_path
    )

    log.info("📊 CV:    AUC=%.3f±%.3f  ACC=%.3f±%.3f  (k=%d)",
             cv_summary["auc_mean"], cv_summary["auc_std"],
             cv_summary["acc_mean"], cv_summary["acc_std"],
             cv_summary["n_splits"])
    log.info("📊 Holdout: AUC=%.3f  ACC=%.3f  (n=%d)",
             holdout["auc"], holdout["acc"], holdout["n_val"])
    log.info("🎉 Training complete.")


if __name__ == "__main__":
    main()
