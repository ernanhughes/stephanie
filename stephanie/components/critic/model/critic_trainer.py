# stephanie/components/critic/agents/critic_trainer.py
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, accuracy_score, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     GroupShuffleSplit, StratifiedKFold)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from stephanie.components.critic.model.trainer import load_dataset
from stephanie.components.critic.reports.training_report import generate_training_reports
from stephanie.utils.hash_utils import hash_list

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

    def __init__(self, cfg, memory, container, logger, run_id):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.run_id = run_id

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
        self.viz_dir = Path(self.cfg.get("viz_dir", f"runs/critic/{self.run_id}/visualizations/"))
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # directionality
        self.directionality: Dict[str, int] = self.cfg.get("directionality", {})

        log.info(
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
    def train(self, X, y, metric_names, groups) -> CriticTrainingResult:
        # ---- 0: Audit (logs which features have NaN/Inf; keep sanitize=False)
        X = self._audit_and_optionally_sanitize_X(X, metric_names, sanitize=False)

        # ---- 1: Directionality correction
        Xc = self._apply_direction(X, metric_names)

        # ---- 2: Feature selection (locked/core)
        Xc, used_names, locked_source = self._select_features(Xc, metric_names)

        # ---- 3: Split
        tr, va = _make_group_holdout(Xc, y, groups)
        Xtr, Xva = Xc[tr], Xc[va]
        ytr, yva = y[tr], y[va]
        groups_tr = groups[tr] if groups is not None else None
        groups_va = groups[va] if groups is not None else None

        # ---- 4: CV + tuning
        model = self._tune(Xtr, ytr, groups_tr)

        # ---- 5: Fit + evaluate
        cv_summary = self._eval_cv(model, Xtr, ytr, groups_tr)
        holdout = self._eval_holdout(model, Xva, yva)

        # ---- 6: Save artifacts (plots are best-effort)
        self._save(model, used_names, cv_summary, holdout, Xva, yva, locked_source,
                       groups_va=groups_va)

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
            log.warning("MetricStore locked-features fetch failed: %s", e)
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
                log.warning(
                    "Locked features missing from dataset (%s): %d → %s",
                    source, len(missing), missing[:10]
                )

            if not keep_idx:
                log.warning("No locked features matched; falling back to %s", "core_only" if self.core_only else "all")
                # fallthrough to core_only / all

            else:
                X_sel = X[:, keep_idx]
                names_sel = [names[i] for i in keep_idx]
                log.info("Using %d locked features from %s", len(names_sel), source)
                return X_sel, names_sel, source

        # No locked list matched → honor core_only if requested
        if self.core_only:
            k = min(8, X.shape[1])
            log.info("core_only=True → using first %d features", k)
            return X[:, :k], names[:k], "core_only"

        # Otherwise keep all
        return X, names, "all"

    # --- Model + evaluation -----------------------------------------------------
    def _tune(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray]):
        """
        Grid-search a robust pipeline (Imputer → Scaler → LogisticRegression)
        using group-aware CV when groups are provided.
        """
        # 1) Build the robust pipeline (imputer handles NaN/Inf; scaler standardizes)
        pipe = self._build_pipeline()  # (imputer, scaler, clf)

        # 2) Hyper-params to search (keep small & stable for tiny datasets)
        #    If you later want to expand, consider C grid or elasticnet l1_ratio.
        param_grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
        }

        # 3) Make CV splitter (group-aware if possible)
        cv, grp = _make_cv(groups, y)

        # 4) Run grid search
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,               # keep deterministic; raise if you want parallelism
            refit=True,
            error_score="raise"     # fail fast if something is wrong
        )
        fit_kwargs = {"groups": grp} if grp is not None else {}
        gs.fit(X, y, **fit_kwargs)

        # 5) Return the best pipeline (already refit on full training folds)
        log.info("Tuning complete: best_params=%s, best_score=%.4f",
                        gs.best_params_, float(gs.best_score_))
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
    def _save(
        self,
        model,
        names: List[str],
        cv: dict,
        holdout: dict,
        Xva: np.ndarray,
        yva: np.ndarray,
        locked_source: str,
        *,
        groups_va: Optional[np.ndarray] = None,  # NEW
    ) -> None:

        # assert trained pipeline matches names
        n_fit = _model_n_features(model)
        if n_fit is not None and n_fit != len(names):
            raise RuntimeError(f"Trainer invariant violated: model fitted on {n_fit} ≠ len(feature_names)={len(names)}")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)

        # 2) Persist feature names (sidecar) for robust recovery
        features_sidecar = Path(self.model_path).with_suffix("").with_name(f"{Path(self.model_path).stem}.features.txt")
        features_sidecar.write_text("\n".join(names), encoding="utf-8")

        # 3) Persist full meta (ALWAYS include feature_names + feature_names_used)
        meta = {
            "feature_names": list(names),          # <- critical
            "feature_names_used": list(names),     # <- compatible alias
            "n_features": int(len(names)),
            "core_only": bool(self.core_only),
            "locked_features_path": str(self.lock_features_path) if self.lock_features_path else None,
            "directionality": dict(self.directionality or {}),
            "cv_summary": cv or {},
            "holdout_summary": holdout or {},
        }
        Path(self.meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # also persist locked features for downstream agents (txt)
        try:
            (self.model_path.parent / "critic.features.txt").write_text(
                "\n".join(names), encoding="utf-8"
            )
        except Exception as e:
            log.warning("Could not write critic.features.txt: %s", e)

        # Training reports are best-effort (never break training)
        try:
            extra_meta = {
                "locked_source": locked_source,
                "run_id": self.run_id,
                "model_path": str(self.model_path),
                "viz_dir": str(self.viz_dir),
            }
            generate_training_reports(
                model=model,
                feature_names=names,
                cv_summary=cv,
                holdout_summary=holdout,
                X_holdout=Xva,
                y_holdout=yva,
                viz_dir=self.viz_dir,
                extra_meta=extra_meta,
            )
        except Exception as e:
            log.warning("CriticTrainer: training reports failed: %s", e)

        # ---- Shadow pack (for CriticInference) ----
        try:
            from stephanie.components.critic.model.shadow import \
                save_shadow_pack

            # Prefer DB-locked kept columns; fall back to used_names
            kept = None
            try:
                kept = self.memory.metrics.get_kept_columns(self.run_id)
            except Exception:
                pass
            if not kept:
                kept = list(names)

            shadow_meta = {
                "cv": cv,
                "holdout": holdout,
                "locked_source": locked_source,
                "run_id": self.run_id,
                "model_path": str(self.model_path),
                "meta_path": str(self.meta_path),
            }

            shadow_path = Path(self.model_path).with_name("critic_shadow.npz")
            save_shadow_pack(
                shadow_path,
                X=Xva,
                y=yva,
                groups=groups_va,
                feature_names=names,
                kept=kept,
                meta=shadow_meta,
            )
            log.info("CriticTrainer: saved shadow pack → %s", shadow_path)
        except Exception as e:
            log.warning("CriticTrainer: shadow pack save skipped: %s", e)

        save_as_candidate = bool(self.cfg.get("save_as_candidate", False))
        model_path = Path(self.cfg.get("candidate_model_path", "models/critic_candidate.joblib")) \
                    if save_as_candidate else self.model_path
        meta_path  = Path(self.cfg.get("candidate_meta_path",  "models/critic_candidate.meta.json")) \
                    if save_as_candidate else self.meta_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        # features sidecar must match the fitted width exactly
        features_sidecar = model_path.with_suffix(model_path.suffix + ".features.txt")
        features_sidecar.write_text("\n".join(names), encoding="utf-8")

        meta = {
            "feature_names": list(names),
            "feature_names_used": list(names),
            "n_features": int(len(names)),
            "core_only": bool(self.core_only),
            "locked_features_path": str(self.lock_features_path) if self.lock_features_path else None,
            "directionality": dict(self.directionality or {}),
            "cv_summary": cv or {},
            "holdout_summary": holdout or {},
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _build_pipeline(self):
        # median is robust; if you prefer zero-fill, use strategy="constant", fill_value=0.0
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),  # keep your current setting; or with_mean=False if sparse
            ("clf",     LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                n_jobs=None,
                class_weight="balanced",   # optional but helpful on small/balanced sets
                random_state=42,
            )),
        ])
 
    def _audit_and_optionally_sanitize_X(self, X: np.ndarray, feature_names: list[str], *, sanitize: bool = False):
        nan_mask  = ~np.isfinite(X)  # True for NaN or ±Inf
        n_bad     = int(nan_mask.sum())
        if n_bad == 0:
            log.info("CriticTrainer: X audit OK (no NaN/Inf)")
            return X

        # column-wise stats
        bad_per_col = nan_mask.sum(axis=0)
        bad_cols = [(feature_names[i] if i < len(feature_names) else f"col_{i}", int(b))
                    for i, b in enumerate(bad_per_col) if b > 0]
        bad_cols.sort(key=lambda kv: kv[1], reverse=True)

        # log top offenders
        top = bad_cols[:10]
        log.warning("CriticTrainer: found %d bad values in X; %d/%d cols affected",
                            n_bad, len(bad_cols), X.shape[1])
        for name, cnt in top:
            log.warning("  - bad[%s] = %d", name, cnt)

        if not sanitize:
            return X

        # Replace non-finites with column medians (or 0.0 if entire col is bad)
        Xc = X.copy()
        for j in range(X.shape[1]):
            col = Xc[:, j]
            mask = ~np.isfinite(col)
            if mask.any():
                # median of finite values
                finite = col[np.isfinite(col)]
                fill = float(np.median(finite)) if finite.size > 0 else 0.0
                col[mask] = fill
                Xc[:, j] = col
        log.info("CriticTrainer: sanitized X (NaN/Inf → median/0.0)")
        return Xc

    def _calculate_data_hash(self, X, y):
        """Calculate a reproducible hash of the data for versioning"""
        data_bytes = X.tobytes() + y.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]
    
def _model_n_features(model) -> int | None:
    # Try to infer from first step that exposes n_features_in_
    try:
        for _, step in model.named_steps.items():
            nfi = getattr(step, "n_features_in_", None)
            if nfi is not None:
                return int(nfi)
        # scikit transforms sometimes keep .statistics_ (imputer)
        imp = model.named_steps.get("imputer")
        if imp is not None and hasattr(imp, "statistics_"):
            return int(imp.statistics_.shape[0])
    except Exception:
        pass
    return None
