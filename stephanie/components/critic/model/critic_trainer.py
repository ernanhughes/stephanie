# stephanie/components/critic/agents/critic_trainer.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     GroupShuffleSplit, StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
# The Shared Trainer (extended, API-compatible)
# ---------------------------------------------------------
class CriticTrainer:
    """
    Stephanie-standard trainer that:
      - Loads tiny_critic dataset
      - Applies directionality
      - Supports feature locking (DB store / file / explicit list), case-insensitive
      - Group-aware CV + tuning
      - Persists model + meta + optional visualizations (best-effort)
    """

    def __init__(self, cfg, memory=None, container=None, logger=None):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger or log

        # paths configurable via YAML
        self.data_path = Path(self.cfg.get("data_path", "data/critic.npz"))
        self.model_path = Path(self.cfg.get("model_path", "models/critic.joblib"))
        self.meta_path = Path(self.cfg.get("meta_path", "models/critic.meta.json"))

        # feature selection
        self.core_only: bool = bool(self.cfg.get("core_only", False))
        self.lock_features_path = (
            Path(self.cfg["lock_features"]) if self.cfg.get("lock_features") else None
        )
        # in-memory names (takes precedence over file)
        self.lock_features_names: List[str] = list(self.cfg.get("lock_features_names") or [])
        # optionally pull from MetricStore (if configured)
        self.lock_from_store: bool = bool(self.cfg.get("lock_features_from_store", True))
        self.lock_store_group_id: Optional[str] = self.cfg.get("lock_store_group_id")  # e.g., run_id
        self.lock_store_meta_key: str = self.cfg.get("lock_store_meta_key", "metric_filter.kept")

        # visualization directory
        self.viz_dir = Path(self.cfg.get("viz_dir", "data/visualizations/critic"))
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # directionality
        self.directionality: Dict[str, int] = self.cfg.get("directionality", {})

        self.logger.info(
            "TinyCriticTrainer initialized",
            extra={"data_path": str(self.data_path), "model_path": str(self.model_path)},
        )

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
    # Core training logic
    # -----------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric_names: List[str],
        groups: Optional[np.ndarray],
    ) -> CriticTrainingResult:

        # ---- 1: Directionality correction
        Xc = self._apply_direction(X, metric_names)

        # ---- 2: Feature selection (locked/core)
        Xc, used_names, locked_source = self._select_features(Xc, metric_names)

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

        # ---- 6: Save artifacts (plots are best-effort)
        self._save(model, used_names, cv_summary, holdout, Xva, yva, locked_source)

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

    # --- Locked features helpers ------------------------------------------------
    def _load_locked_from_store(self) -> Optional[List[str]]:
        """
        Try to load kept/locked features from MetricStore.meta for a group/run_id.
        Expects the kept list under meta[self.lock_store_meta_key].
        """
        if not self.lock_from_store or not self.container or not self.lock_store_group_id:
            return None
        try:
            metric_store = self.container.get("metrics")
            if not metric_store:
                return None
            # compatible with earlier helpers we added:
            g = metric_store.get_or_create_group(self.lock_store_group_id)
            meta = dict(getattr(g, "meta", {}) or {})
            # Examples of accepted keys:
            #   "metric_filter.kept": [...]
            #   "metric_filter": {"kept": [...]}
            if self.lock_store_meta_key in meta and isinstance(meta[self.lock_store_meta_key], list):
                return list(meta[self.lock_store_meta_key])
            mf = meta.get("metric_filter")
            if isinstance(mf, dict) and isinstance(mf.get("kept"), list):
                return list(mf["kept"])
        except Exception as e:
            self.logger.warning("MetricStore locked-features fetch failed: %s", e)
        return None

    def _resolve_locked_names(
        self, all_feature_names: List[str]
    ) -> Tuple[Optional[List[str]], str]:
        """
        Decide which locked names to use, in priority order:
          1) in-memory cfg.lock_features_names
          2) database MetricStore (kept columns)
          3) lock_features file
          4) None (fallback to core_only or all)
        Returns (names_or_None, source_label)
        """
        # 1) explicit list in cfg
        if self.lock_features_names:
            return self.lock_features_names, "cfg.lock_features_names"

        # 2) DB store
        store_names = self._load_locked_from_store()
        if store_names:
            return store_names, f"metric_store:{self.lock_store_group_id}"

        # 3) file
        if self.lock_features_path and self.lock_features_path.exists():
            names = [
                ln.strip()
                for ln in self.lock_features_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            if names:
                return names, f"file:{self.lock_features_path}"

        # 4) none
        return None, "none"

    @staticmethod
    def _case_insensitive_map(names: List[str]) -> Dict[str, str]:
        """Map lowercase name -> original canonical name."""
        out = {}
        for n in names:
            key = n.lower()
            if key not in out:
                out[key] = n
        return out

    def _select_features(self, X, names):
        """
        Returns:
          X_sel, names_sel, locked_source_label
        """
        # Decide locked names
        locked_names_raw, source = self._resolve_locked_names(names)

        if locked_names_raw:
            # case-insensitive matching to dataset names
            actual_map = self._case_insensitive_map(names)
            keep_idx = []
            missing = []
            resolved_names = []
            for ln in locked_names_raw:
                cand = actual_map.get(ln.lower())
                if cand is None:
                    missing.append(ln)
                    continue
                resolved_names.append(cand)

            name_to_idx = {n: i for i, n in enumerate(names)}
            keep_idx = [name_to_idx[n] for n in resolved_names if n in name_to_idx]

            if missing:
                self.logger.warning(
                    "Locked features missing from dataset (%s): %d → %s",
                    source, len(missing), missing[:10]
                )

            if not keep_idx:
                self.logger.warning("No locked features matched; falling back to %s", "core_only" if self.core_only else "all")
                # fallthrough to core_only / all

            else:
                X_sel = X[:, keep_idx]
                names_sel = [names[i] for i in keep_idx]
                self.logger.info("Using %d locked features from %s", len(names_sel), source)
                return X_sel, names_sel, source

        # No locked list matched → honor core_only if requested
        if self.core_only:
            k = min(8, X.shape[1])
            self.logger.info("core_only=True → using first %d features", k)
            return X[:, :k], names[:k], "core_only"

        # Otherwise keep all
        return X, names, "all"

    # --- Model + evaluation -----------------------------------------------------
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

    # --- Persistence + best-effort plots ---------------------------------------
    def _save(self, model, names, cv, holdout, Xva, yva, locked_source: str):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)

        meta = {
            "feature_names": names,
            "core_only": self.core_only,
            "locked_source": locked_source,
            "directionality": self.directionality,
            "cv": cv,
            "holdout": holdout,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

        # also persist locked features for downstream agents (txt)
        try:
            (self.model_path.parent / "critic.features.txt").write_text(
                "\n".join(names), encoding="utf-8"
            )
        except Exception as e:
            self.logger.warning("Could not write critic.features.txt: %s", e)

        # Plots are best-effort (never break training)
        try:
            self._plot_coefficients(model, names, self.viz_dir / "coef_importance.png")
        except Exception as e:
            self.logger.warning("coef plot failed: %s", e)

        try:
            self._plot_holdout_cm(model, Xva, yva, self.viz_dir / "holdout_confusion.png")
        except Exception as e:
            self.logger.warning("confusion plot failed: %s", e)

        try:
            self._plot_holdout_roc(model, Xva, yva, self.viz_dir / "holdout_roc.png")
        except Exception as e:
            self.logger.warning("roc plot failed: %s", e)

    # --- Plot helpers -----------------------------------------------------------
    def _plot_coefficients(self, model, feature_names: List[str], out_path: Path, top_k: int = 30):
        # unwrap pipeline to logisticregression step
        try:
            lr = None
            for step in getattr(model, "steps", []):
                if hasattr(step[1], "coef_"):
                    lr = step[1]
            if lr is None and hasattr(model, "coef_"):
                lr = model
            if lr is None:
                raise ValueError("no linear model with coef_ found")
            coefs = lr.coef_.ravel()
        except Exception:
            # maybe liblinear returns classes_[1] coef only when fitted; ensure we can extract
            raise

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

    def _plot_holdout_cm(self, model, X: np.ndarray, y: np.ndarray, out_path: Path):
        if len(y) == 0 or len(np.unique(y)) < 2:
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

    def _plot_holdout_roc(self, model, X: np.ndarray, y: np.ndarray, out_path: Path):
        if len(y) == 0 or len(np.unique(y)) < 2:
            return
        p = model.predict_proba(X)[:, 1]
        plt.figure(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y, p)
        plt.title("TinyCritic | Holdout ROC")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
