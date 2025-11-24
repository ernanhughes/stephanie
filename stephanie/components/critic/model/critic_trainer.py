# stephanie/components/critic/agents/critic_trainer.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import (
    GroupKFold, StratifiedKFold, GroupShuffleSplit, GridSearchCV
)

from stephanie.components.critic.model.trainer import load_dataset


log = logging.getLogger(__name__)


# ---------------------------------------------------------
# Dataclass: tiny critic training results
# ---------------------------------------------------------
@dataclass
class CriticTrainingResult:
    model_path: str
    meta_path: str
    cv_summary: dict
    holdout_summary: dict


# ---------------------------------------------------------
# Utility splitters
# ---------------------------------------------------------
def _make_group_holdout(X, y, groups, test_size=0.2, seed=42):
    if groups is not None:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        tr, va = next(splitter.split(X, y, groups))
        return tr, va
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(len(idx) * (1.0 - test_size))
    return idx[:cut], idx[cut:]


def _make_cv(groups, y, n_splits=5):
    if groups is not None and len(np.unique(groups)) >= n_splits:
        return GroupKFold(n_splits=n_splits), groups
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), None


# ---------------------------------------------------------
# The Shared Trainer
# ---------------------------------------------------------
class CriticTrainer:
    """
    A Stephanie-standard trainer that:
      - Loads tiny_critic dataset
      - Performs directionality correction
      - Does group-aware CV + tuning
      - Saves model + meta
      - Returns structured stats for dashboards and agents

    This class replaces legacy trainer.py but uses the same internals.
    """

    def __init__(self, cfg, memory=None, container=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger or log

        # paths configurable via YAML
        self.data_path = Path(cfg.get("data_path", "data/critic.npz"))
        self.model_path = Path(cfg.get("model_path", "models/critic.joblib"))
        self.meta_path = Path(cfg.get("meta_path", "models/critic.meta.json"))
        self.core_only = bool(cfg.get("core_only", True))
        self.lock_features_path = (
            Path(cfg["lock_features"]) if cfg.get("lock_features") else None
        )

        # visualization directory
        self.viz_dir = Path(cfg.get("viz_dir", "data/visualizations/tiny_critic"))
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # directionality (same as legacy)
        self.directionality = cfg.get("directionality", {})

        self.logger.info("TinyCriticTrainer initialized", extra={
            "data_path": str(self.data_path),
            "model_path": str(self.model_path)
        })

    # -----------------------------------------------------
    # API ENTRY: Train from NPZ dataset (used by CLI + agent)
    # -----------------------------------------------------
    def train_from_dataset(self, path: Optional[str | Path] = None) -> CriticTrainingResult:
        """
        Load dataset, train model, save results.
        """
        data_path = Path(path or self.data_path)
        X, y, metric_names, groups = load_dataset(data_path)

        return self.train(X, y, metric_names, groups)

    # -----------------------------------------------------
    # Core training logic (pure function)
    # -----------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric_names: list[str],
        groups: Optional[np.ndarray],
    ) -> CriticTrainingResult:

        # ---- 1: Directionality correction
        Xc = self._apply_direction(X, metric_names)

        # ---- 2: Feature selection (core only or locked list)
        Xc, used_names = self._select_features(Xc, metric_names)

        # ---- 3: Split
        tr, va = _make_group_holdout(Xc, y, groups)
        Xtr, Xva = Xc[tr], Xc[va]
        ytr, yva = y[tr], y[va]
        groups_tr = groups[tr] if groups is not None else None

        # ---- 4: CV + tuning
        model = self._tune(Xtr, ytr, groups_tr)

        # ---- 5: Fit + evaluate
        cv_summary = self._eval_cv(model, Xtr, ytr, groups_tr)
        holdout = self._eval_holdout(model, Xva, yva)

        # ---- 6: Save artifacts
        self._save(model, used_names, cv_summary, holdout)

        # ---- return structured result
        return CriticTrainingResult(
            model_path=str(self.model_path),
            meta_path=str(self.meta_path),
            cv_summary=cv_summary,
            holdout_summary=holdout,
        )

    # -----------------------------------------------------
    # Internals
    # -----------------------------------------------------
    def _apply_direction(self, X, feature_names):
        Xc = X.copy()
        for i, n in enumerate(feature_names):
            d = self.directionality.get(n)
            if d == -1:
                Xc[:, i] = -Xc[:, i]
        return Xc

    def _select_features(self, X, names):
        if self.lock_features_path and self.lock_features_path.exists():
            locked = [
                ln.strip()
                for ln in self.lock_features_path.read_text().splitlines()
                if ln.strip()
            ]
            keep = [names.index(k) for k in locked if k in names]
            return X[:, keep], locked

        if self.core_only:
            keep = list(range(min(8, X.shape[1])))
            return X[:, keep], names[:len(keep)]

        return X, names

    def _tune(self, X, y, groups):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                random_state=42,
            ),
        )
        params = {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}

        cv, grp = _make_cv(groups, y)
        gs = GridSearchCV(
            pipe,
            params,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1
        )
        gs.fit(X, y, **({"groups": grp} if grp is not None else {}))
        return gs.best_estimator_

    def _eval_cv(self, model, X, y, groups):
        cv, grp = _make_cv(groups, y)
        aucs, accs = [], []
        for tr, va in cv.split(X, y, groups=grp):
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[va])[:, 1]
            aucs.append(roc_auc_score(y[va], p))
            accs.append(accuracy_score(y[va], (p >= 0.5).astype(int)))
        return {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
        }

    def _eval_holdout(self, model, Xva, yva):
        p = model.predict_proba(Xva)[:, 1]
        yhat = (p >= 0.5).astype(int)
        return {
            "auc": float(roc_auc_score(yva, p)) if len(np.unique(yva)) > 1 else 0.0,
            "acc": float(accuracy_score(yva, yhat)),
            "n": int(len(yva)),
        }

    def _save(self, model, names, cv, holdout):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)

        meta = {
            "feature_names": names,
            "core_only": self.core_only,
            "directionality": self.directionality,
            "cv": cv,
            "holdout": holdout,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

        # ---- Confusion Matrix
        # (Small viz but helpful)
        try:
            plt.figure(figsize=(5, 4))
            cm = confusion_matrix(
                np.array([1, 0]),
                np.array([1, 0])  # static example; replace with real if needed
            )
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("TinyCritic (example)")
            plt.tight_layout()
            plt.savefig(self.viz_dir / "confusion_matrix.png")
            plt.close()
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")

