# stephanie/agents/maintenance/risk_trainer.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
# Optional deps
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
from stephanie.agents.base_agent import BaseAgent

# ---- Small helpers ---------------------------------------------------------

def _safe_str(x: Any) -> str:
    try:
        return str(x) if x is not None else ""
    except Exception:
        return ""

def _pii_strip(s: str) -> str:
    # Minimal PII cleanup (align with predictor)
    import re
    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")
    s = EMAIL_RE.sub("<EMAIL>", s or "")
    s = PHONE_RE.sub("<PHONE>", s)
    return s

def _ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (binary)."""
    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    m = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        w = float(np.mean(mask))
        ece += w * abs(acc - conf)
    return float(ece)

def _percentile_thresholds(p: np.ndarray, lo_pct: float = 20.0, hi_pct: float = 80.0) -> Tuple[float, float]:
    lo = float(np.percentile(p, lo_pct))
    hi = float(np.percentile(p, hi_pct))
    hi = max(hi, lo + 0.01)
    return max(0.0, min(lo, 0.95)), max(0.01, min(hi, 0.99))

# ---- Fallback mini featurizer (matches predictor contract) -----------------

DEFAULT_FEATURE_ORDER = [
    "q_len", "ctx_len", "overlap_ratio",
    "ner_count", "num_tokens_est",
    "coverage_gap", "prior_max_energy_ema",
]

def _row_features(row: pd.Series) -> Dict[str, float]:
    q = _pii_strip(_safe_str(row.get("question", row.get("goal_text", ""))))
    c = _pii_strip(_safe_str(row.get("context", "")))
    q_tokens = q.split()
    c_tokens = c.split()
    inter = len(set(q_tokens) & set(c_tokens))
    denom = max(1, len(set(q_tokens)))
    overlap = inter / denom
    return {
        "q_len": float(len(q)),
        "ctx_len": float(len(c)),
        "overlap_ratio": float(overlap),
        "ner_count": float(sum(w.istitle() for w in q_tokens)),
        "num_tokens_est": float(len(q_tokens) + len(c_tokens)),
        "coverage_gap": float(1.0 - overlap),
        "prior_max_energy_ema": float(row.get("prior_max_energy_ema", 0.25)),
    }

def _ensure_features(df: pd.DataFrame, feature_order: List[str]) -> pd.DataFrame:
    # If columns already present, keep; else derive.
    missing = [f for f in feature_order if f not in df.columns]
    if missing:
        feats = [ _row_features(r) for _, r in df.iterrows() ]
        feat_df = pd.DataFrame(feats)
        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values
    # Guarantee column presence
    for f in feature_order:
        if f not in df.columns:
            df[f] = 0.0
    return df

def _pick_label(df: pd.DataFrame) -> np.ndarray:
    """
    Choose binary label column in order of preference:
      1) 'label' (0/1)
      2) 'hallucination' (bool/int)
      3) 'max_energy' (threshold into binary)
    """
    if "label" in df.columns:
        return df["label"].astype(int).values
    if "hallucination" in df.columns:
        return (df["hallucination"].astype(int).values > 0).astype(int)
    if "max_energy" in df.columns:
        # Threshold at 0.8 if not already labeled
        return (df["max_energy"].astype(float).values >= 0.8).astype(int)
    raise ValueError("No binary label column found (expected 'label' or 'hallucination' or 'max_energy').")

# ---- Trainer ---------------------------------------------------------------

@dataclass
class RiskTrainer:
    """
    Trains a calibrated risk classifier on a risk dataset (DataFrame),
    then computes per-domain thresholds and persists a model bundle.

    - Model: XGBoost (if available) or GradientBoostingClassifier
    - Calibration: Isotonic via CalibratedClassifierCV
    - Thresholds: per-domain low/high via percentiles on validation proba
    """
    cfg: Dict[str, Any]
    memory: Any
    container: Any
    logger: Any

    feature_order: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_ORDER))

    def _memcube(self):
        # Optional: grab memcube-like service if present
        try:
            return self.container.get("memcube")
        except Exception:
            return None

    def _out_dir(self) -> Path:
        return Path(self.cfg.get("out_dir", "models/risk")).absolute()

    def _domains(self, df: pd.DataFrame) -> List[str]:
        if "domain" in df.columns:
            return sorted([str(x) for x in df["domain"].fillna("general").unique().tolist()])
        return ["general"]

    # Convenience (if you prefer calling from parquet path)
    def train_from_parquet(self, parquet_path: str | Path) -> Dict[str, Any]:
        df = pd.read_parquet(parquet_path)
        return self.train_dataframe(df)

    # === The method you asked for ===
    def train_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.logger.log("RiskTrainerStart", {"rows": int(df.shape[0]), "cols": int(df.shape[1])})

        # 1) Ensure features + label + domain
        df = df.copy()
        df = _ensure_features(df, self.feature_order)
        if "domain" not in df.columns:
            df["domain"] = "general"
        domains = self._domains(df)

        y = _pick_label(df).astype(int)
        X = df[self.feature_order].astype(float).values
        dom_series = df["domain"].astype(str)

        # 2) Split (stratify by y; domains used for per-domain eval later)
        test_size = float(self.cfg.get("test_size", 0.2))
        seed = int(self.cfg.get("seed", 42))
        Xtr, Xva, ytr, yva, dtr, dva = train_test_split(
            X, y, dom_series.values, test_size=test_size, random_state=seed, stratify=y
        )

        # 3) Fit base classifier
        if _HAS_XGB:
            xgb_cfg = self.cfg.get("xgb", {}) or {}
            clf_base = xgb.XGBClassifier(
                n_estimators=int(xgb_cfg.get("n_estimators", 300)),
                max_depth=int(xgb_cfg.get("max_depth", 5)),
                learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
                subsample=float(xgb_cfg.get("subsample", 0.9)),
                colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
                reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=int(xgb_cfg.get("n_jobs", 4)),
            )
        else:
            # Reasonable fallback if xgboost isn't available
            clf_base = GradientBoostingClassifier(
                learning_rate=0.05, n_estimators=300, max_depth=3, random_state=seed
            )

        clf_base.fit(Xtr, ytr)

        # 4) Isotonic calibration on validation fold
        calib = CalibratedClassifierCV(base_estimator=clf_base, cv="prefit", method="isotonic")
        calib.fit(Xva, yva)

        # 5) Global metrics on validation
        proba_va = calib.predict_proba(Xva)[:, 1]
        roc = float(roc_auc_score(yva, proba_va)) if len(np.unique(yva)) > 1 else float("nan")
        pr  = float(average_precision_score(yva, proba_va)) if len(np.unique(yva)) > 1 else float("nan")
        ece = _ece_binary(yva, proba_va, n_bins=int(self.cfg.get("ece_bins", 15)))

        # 6) Per-domain thresholds on validation
        thresholds: Dict[str, Dict[str, float]] = {}
        default_low  = float(self.cfg.get("thresholds", {}).get("default_low", 0.20))
        default_high = float(self.cfg.get("thresholds", {}).get("default_high", 0.60))

        for dom in domains:
            mask = (dva == dom)
            if not np.any(mask):
                thresholds[dom] = {"low_threshold": default_low, "high_threshold": default_high, "count": 0}
                continue
            p = proba_va[mask]
            lo, hi = _percentile_thresholds(p, lo_pct=float(self.cfg.get("lo_pct", 20.0)),
                                               hi_pct=float(self.cfg.get("hi_pct", 80.0)))
            thresholds[dom] = {
                "low_threshold": float(lo),
                "high_threshold": float(hi),
                "count": int(mask.sum()),
            }

        # 7) Persist model bundle
        out_dir = self._out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = out_dir / "bundle.joblib"
        bundle = {
            "clf": calib,                       # calibrated model
            "feature_names": self.feature_order,
            "version": "risk-bundle.v1",
        }
        dump(bundle, bundle_path)

        # 8) Persist calibration
        memcube = self._memcube()
        for dom, rec in thresholds.items():
            payload = {
                "domain": dom,
                "low_threshold": rec["low_threshold"],
                "high_threshold": rec["high_threshold"],
                "ece": ece,
                "sample_count": rec["count"],
            }
            if memcube is not None and hasattr(memcube, "store_calibration"):
                try:
                    # async interface in runtime; here we allow sync fallback
                    maybe_coro = memcube.store_calibration("risk", payload)
                    if hasattr(maybe_coro, "__await__"):
                        import asyncio
                        asyncio.get_event_loop().run_until_complete(maybe_coro)
                except Exception:
                    pass

        # Always write a local copy for transparency
        with open(out_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump({"thresholds": thresholds, "ece": ece}, f, indent=2)

        # 9) Return training metadata
        meta = {
            "status": "ok",
            "rows": int(df.shape[0]),
            "feature_names": list(self.feature_order),
            "domains": domains,
            "val_roc_auc": roc,
            "val_pr_auc": pr,
            "val_ece": ece,
            "thresholds": thresholds,
            "bundle_path": str(bundle_path),
            "model_type": "xgb_isotonic" if _HAS_XGB else "gb_isotonic",
            "seed": seed,
            "test_size": test_size,
        }
        self.logger.log("RiskTrainerDone", meta)
        return meta


def _maybe_build_dataset(
    run_dir: Optional[str],
    out_parquet: str,
    energy_thresh: float,
    logger,
) -> None:
    """
    If a GAP run directory is provided, (re)build the risk dataset parquet.
    Falls back gracefully if import path differs (e.g., local scripts layout).
    """
    if not run_dir:
        return

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Preferred: call the builder function directly (no subprocess)
        from scripts.build_risk_dataset import build_from_run

        logger.log(
            "RiskDatasetBuildStart",
            {
                "run_dir": str(run_dir),
                "out": str(out_parquet),
                "energy_thresh": energy_thresh,
            },
        )
        build_from_run(
            Path(run_dir), out_path, label_energy_thresh=energy_thresh
        )
        logger.log("RiskDatasetBuildDone", {"rows_path": str(out_parquet)})
    except Exception as e:
        # Fallback no-op: leave existing dataset as-is; agent won’t crash
        logger.log(
            "RiskDatasetBuildFailed",
            {"error": str(e), "run_dir": str(run_dir)},
        )


class RiskTrainerAgent(BaseAgent):
    """
    Maintenance agent that trains the Risk Predictor (hallucination risk)
    from a parquet dataset (optionally built from a GAP run directory).

    Config (example):
      risk_trainer_agent:
        run_dir: ${hydra:runtime.cwd}/data/gap_runs/7462
        data_path: ${hydra:runtime.cwd}/reports/risk_dataset.parquet
        energy_thresh: 0.80
        trainer:
          out_dir: ${hydra:runtime.cwd}/models/risk
          test_size: 0.2
          seed: 42
          xgb:
            max_depth: 5
            n_estimators: 300
            learning_rate: 0.05
          thresholds:
            default_low: 0.20
            default_high: 0.60
        output_key: risk_training
        hot_reload_service: true
    """

    def __init__(self, cfg: dict, memory, container: ServiceContainer, logger):
        super().__init__(cfg, memory, container, logger)
        self.run_dir: str = cfg.get("run_dir", "")
        self.data_path: str = cfg.get("data_path", "reports/risk_dataset.parquet")
        self.energy_thresh: float = float(cfg.get("energy_thresh", 0.80))
        self.trainer_cfg: dict = cfg.get("trainer", {}) or {}
        self.output_key: str = cfg.get("output_key", "risk_training")
        self.hot_reload_service: bool = bool(cfg.get("hot_reload_service", True))

        self.trainer = RiskTrainer(
            cfg=self.trainer_cfg, memory=memory, container=container, logger=logger
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Ensure dataset exists (or build from GAP run)
        _maybe_build_dataset(
            run_dir=self.run_dir,
            out_parquet=self.data_path,
            energy_thresh=self.energy_thresh,
            logger=self.logger,
        )

        # 2) Load samples from parquet (agent-side)
        #    If your RiskTrainer already has train_from_parquet(), you can skip
        #    this and call trainer.train_from_parquet(self.data_path) instead.
        df = pd.read_parquet(self.data_path)

        # Optional quick sanity logs
        self.logger.log("RiskDatasetLoaded", {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "path": str(self.data_path),
        })

        # 3) Train model (XGB + Isotonic + per-domain sweep)
        #    The RiskTrainer should accept a DataFrame OR provide a helper like
        #    train_from_parquet(). Below we call the DataFrame path explicitly.
        meta = self.trainer.train_dataframe(df)

        # 4) (Optional) Hot-reload the runtime service so scorers pick up the new model
        if self.hot_reload_service:
            try:
                svc = self.container.get("risk_predictor")
                if hasattr(svc, "reload_bundle") and callable(svc.reload_bundle):
                    svc.reload_bundle()
                    self.logger.log("RiskServiceReloaded", {"ok": True})
                else:
                    self.logger.log("RiskServiceReloadSkipped", {"reason": "no reload_bundle()"})
            except Exception as e:
                self.logger.log("RiskServiceReloadFailed", {"error": str(e)})

        # 3) Return stats in context (mirrors your other agents)
        context[self.output_key] = {
            "status": "ok",
            "dataset_path": str(self.data_path),
            "model_out_dir": self.trainer_cfg.get("out_dir", "models/risk"),
            "meta": meta,
        }
        return context
