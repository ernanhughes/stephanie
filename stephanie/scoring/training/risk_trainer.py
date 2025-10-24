# stephanie/scoring/training/risk_trainer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from stephanie.scoring.training.base_trainer import BaseTrainer


@dataclass
class RiskTrainerConfig:
    data_path: str = "reports/risk_dataset.parquet"
    out_dir: str = "models/risk"
    # XGB params
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.03
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    random_state: int = 42
    test_size: float = 0.2
    # domain thresholds sweep
    gate_low_percentile: float = 0.20
    gate_high_percentile: float = 0.80
    # feature filters
    use_scm: bool = True     # keep columns prefixed "scm."
    use_delta: bool = True   # keep columns prefixed "delta."
    # column names
    label_col: str = "y"
    domain_col: str = "domain"


class RiskTrainer(BaseTrainer):
    """
    Trainer for hallucination risk predictor.
    - XGBoost binary classifier + Isotonic calibration
    - One-hot domain feature
    - Optional per-domain gate thresholds (low/high) for routing
    Outputs a single bundle.joblib consumed by RiskPredictorService.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = RiskTrainerConfig(**cfg) if isinstance(cfg, dict) else cfg

    # ---------- public API ----------
    def train(self, samples=None, dimension="risk"):
        """
        For parity with other trainers, `samples` is unused:
        the risk model trains from a parquet dataset produced by build_risk_dataset.py
        """
        data_path = Path(self.cfg.data_path)
        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(data_path)

        # Sanity: ensure minimal columns
        missing = [c for c in [self.cfg.label_col, self.cfg.domain_col] if c not in df.columns]
        if missing:
            raise ValueError(f"RiskTrainer: dataset missing columns: {missing}")

        X, y, feature_names, domain_encoder = self._make_matrix(df)

        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=self.cfg.test_size, random_state=self.cfg.random_state, stratify=y
        )

        clf = XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_lambda=self.cfg.reg_lambda,
            tree_method="hist",
            eval_metric="logloss",
            random_state=self.cfg.random_state,
        )
        clf.fit(Xtr, ytr)

        # Isotonic calibration
        p_va_raw = clf.predict_proba(Xva)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_va_raw, yva)
        p_va = iso.predict(p_va_raw)

        # Metrics
        val_auc = float(roc_auc_score(yva, p_va))
        val_ap = float(average_precision_score(yva, p_va))
        val_logloss = float(log_loss(yva, np.clip(p_va, 1e-6, 1 - 1e-6)))
        prevalence = float(y.mean())

        # Per-domain gate thresholds (low/high) using percentiles of calibrated p
        thresholds = self._compute_domain_thresholds(df, domain_encoder, clf, iso)

        # Persist bundle
        bundle = {
            "version": "risk_xgb_isotonic_v1",
            "clf": clf,
            "iso": iso,
            "feature_names": feature_names,
            "domain_encoder": domain_encoder,
            "thresholds": thresholds,  # {"domain": {"low":..,"high":..}}
            "metrics": {
                "val_auc": val_auc,
                "val_ap": val_ap,
                "val_logloss": val_logloss,
                "prevalence": prevalence,
                "n_train": int(len(ytr)),
                "n_val": int(len(yva)),
            },
        }
        joblib.dump(bundle, out_dir / "bundle.joblib")
        (out_dir / "metrics.json").write_text(json.dumps(bundle["metrics"], indent=2))

        # TrainingStats (lightweight)
        try:
            self.memory.training_stats.add_from_result(
                stats={
                    "auc": val_auc,
                    "ap": val_ap,
                    "logloss": val_logloss,
                    "prevalence": prevalence,
                },
                model_type="risk",
                target_type="binary",
                dimension=dimension,
                version="xgb_isotonic_v1",
                embedding_type="tabular",
                config={
                    "n_estimators": self.cfg.n_estimators,
                    "max_depth": self.cfg.max_depth,
                    "lr": self.cfg.learning_rate,
                    "subsample": self.cfg.subsample,
                    "colsample_bytree": self.cfg.colsample_bytree,
                    "reg_lambda": self.cfg.reg_lambda,
                    "test_size": self.cfg.test_size,
                },
                sample_count=len(df),
                start_time=datetime.now(),
            )
        except Exception:
            pass

        meta = {
            "model_type": "risk",
            "version": "xgb_isotonic_v1",
            "feature_count": len(feature_names),
            "domain_count": int(len(domain_encoder.categories_[0])) if hasattr(domain_encoder, "categories_") else 0,
            **bundle["metrics"],
        }
        self._save_meta_file(meta, dimension)
        self.logger.log("RiskTrainingComplete", meta)
        return meta

    # ---------- helpers ----------
    def _make_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], OneHotEncoder]:
        # Numeric features
        num_cols = []
        if self.cfg.use_delta:
            num_cols += [c for c in df.columns if c.startswith("delta.")]
        if self.cfg.use_scm:
            num_cols += [c for c in df.columns if c.startswith("scm.")]
        # always include lengths if present
        for c in ("len_goal", "len_out"):
            if c in df.columns:
                num_cols.append(c)

        num_cols = sorted(set(num_cols))
        Xnum = df[num_cols].fillna(0.0).values.astype(np.float32)

        # One-hot domain
        oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        Xdom = oh.fit_transform(df[[self.cfg.domain_col]])
        dom_names = [f"dom:{x}" for x in oh.get_feature_names_out([self.cfg.domain_col])]
        feature_names = num_cols + dom_names

        X = np.hstack([Xnum, Xdom.astype(np.float32)])
        y = df[self.cfg.label_col].astype(int).values

        return X, y, feature_names, oh

    def _compute_domain_thresholds(
        self,
        df: pd.DataFrame,
        domain_encoder: OneHotEncoder,
        clf: XGBClassifier,
        iso: IsotonicRegression,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute low/high gates per domain by evaluating calibrated probabilities
        on validation-like slices. If dataset is small per domain, falls back to global.
        """
        domains = list(sorted(df[self.cfg.domain_col].astype(str).unique()))
        thresholds: Dict[str, Dict[str, float]] = {}

        # Build features once for speed
        X, y, feature_names, _ = self._make_matrix(df)
        p_cal = iso.predict(clf.predict_proba(X)[:, 1])

        for dom in domains:
            mask = (df[self.cfg.domain_col].astype(str) == dom).values
            if mask.sum() < 30:
                # too small; use global thresholds
                low = float(np.quantile(p_cal, self.cfg.gate_low_percentile))
                high = float(np.quantile(p_cal, self.cfg.gate_high_percentile))
            else:
                low = float(np.quantile(p_cal[mask], self.cfg.gate_low_percentile))
                high = float(np.quantile(p_cal[mask], self.cfg.gate_high_percentile))
            thresholds[dom] = {"low": low, "high": high}
        return thresholds
