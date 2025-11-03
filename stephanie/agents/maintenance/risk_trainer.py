# stephanie/agents/maintenance/risk_trainer.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Optional deps
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
from stephanie.agents.base_agent import BaseAgent

# ---- Small helpers ---------------------------------------------------------


HRM_DIMS  = ["reasoning","knowledge","clarity","faithfulness","coverage"]

def _ensure_domain(df: pd.DataFrame) -> pd.DataFrame:
    if "domain" in df.columns:
        return df
    # best effort from provenance.dimension or simple heuristics
    if "dimension" in df.columns:
        return df.rename(columns={"dimension": "domain"})
    # fallback: everything "general"
    df = df.copy()
    df["domain"] = "general"
    return df

def _synth_disagreement_label(
    df: pd.DataFrame,
    per_dim_delta: float = 0.25,
    multi_dim_min: int = 2,
) -> pd.Series:
    """
    Label=1 if HRM and Tiny disagree enough across several dims.
    """
    diffs = []
    for d in HRM_DIMS:
        h = f"HRM.{d}"
        t = f"Tiny.{d}"
        if h in df.columns and t in df.columns:
            diffs.append((df[h] - df[t]).abs())
    if not diffs:
        # no basis — return all zeros (trainer will raise if still unlabeled)
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    D = pd.concat(diffs, axis=1)  # shape [N, <=5]
    hits = (D >= per_dim_delta).sum(axis=1)              # how many dims disagree meaningfully
    lbl  = (hits >= multi_dim_min).astype(int)
    return lbl

def _pick_or_build_label(
    df: pd.DataFrame,
    energy_thresh: float = 0.80,
    disagree_delta: float = 0.25,
    disagree_dims_min: int = 2,
) -> pd.Series:
    """
    Priority:
      1) explicit boolean/0-1 'label'
      2) 'hallucination' truthy/0-1
      3) 'max_energy' >= energy_thresh
      4) synth label from HRM/Tiny disagreement across dimensions
    Returns a 0/1 series (int).
    """
    if "label" in df.columns:
        s = df["label"]
        # normalize
        if s.dtype == bool:
            return s.astype(int)
        return (s.astype(float) >= 0.5).astype(int)

    if "hallucination" in df.columns:
        s = df["hallucination"]
        if s.dtype == bool:
            return s.astype(int)
        return (s.astype(float) >= 0.5).astype(int)

    if "max_energy" in df.columns:
        return (df["max_energy"].astype(float) >= float(energy_thresh)).astype(int)

    # Last resort: synthesize from disagreement
    return _synth_disagreement_label(
        df,
        per_dim_delta=float(disagree_delta),
        multi_dim_min=int(disagree_dims_min),
    )

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

    def train_from_parquet(self, parquet_path: str | Path) -> Dict[str, Any]:
        df = pd.read_parquet(parquet_path)
        return self.train_dataframe(df)

    def train_dataframe(self, df: pd.DataFrame) -> dict:
        # -------------------------
        # 0) Basic hygiene
        # -------------------------
        df = df.copy()
        df = _ensure_domain(df)

        # -------------------------
        # 1) Features (use what you have)
        # -------------------------
        # Prefer your engineered columns: lengths + deltas
        delta_cols = [c for c in df.columns if c.startswith("delta.")]
        base_feat_cols = ["len_goal", "len_out"] + delta_cols
        # Fill missing numeric cols with 0.0 so we never crash
        for c in base_feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        X = df[base_feat_cols].astype(float)

        # -------------------------
        # 2) Labels (robust synthesis)
        # -------------------------
        # Try explicit y/label/hallucination first
        y = None
        if "label" in df.columns:
            y = (df["label"].astype(float) >= 0.5).astype(int)
        elif "hallucination" in df.columns:
            y = (df["hallucination"].astype(float) >= 0.5).astype(int)
        elif "y" in df.columns:
            y = df["y"].astype(int)

        # If label is missing or degenerate (all the same), synthesize from deltas
        def _synth_from_deltas(frame: pd.DataFrame, pos_rate_target: float = 0.20) -> pd.Series:
            # Score of “disagreement energy”: max |delta| across the 5 dims
            if not delta_cols:
                # If no deltas, fallback to zeros (trainer will throw later)
                return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)

            D = frame[delta_cols].astype(float)
            # use max absolute deviation as risk proxy
            score = D.abs().max(axis=1)

            # Per-domain percentile cut to hit target positive rate in each domain
            dom = frame["domain"].astype(str).fillna("general")
            labels = np.zeros(len(frame), dtype=int)
            for dd in dom.unique():
                mask = (dom == dd).values
                if not np.any(mask):
                    continue
                cutoff = np.quantile(score[mask].values, 1.0 - pos_rate_target)
                labels[mask] = (score[mask].values >= cutoff).astype(int)
            return pd.Series(labels, index=frame.index)

        if y is None or y.nunique() < 2 or y.sum() == 0:
            # synthesize ~20% positives per domain (configurable via cfg)
            target_pos_rate = float(self.cfg.get("target_pos_rate", 0.20))
            y = _synth_from_deltas(df, pos_rate_target=target_pos_rate)

        # Final sanity: ensure we have both classes
        if y.nunique() < 2 or y.sum() == 0:
            raise ValueError(
                "Unable to construct a usable binary label from dataset. "
                "Consider adjusting target_pos_rate (e.g., 0.30) or regenerating the dataset."
            )

        # -------------------------
        # 3) Split
        # -------------------------
        test_size = float(self.cfg.get("test_size", 0.20))
        seed = int(self.cfg.get("seed", 42))
        dom = df["domain"].fillna("general").astype(str).values

        X_train, X_val, y_train, y_val, d_train, d_val = train_test_split(
            X.values, y.values, dom, test_size=test_size, random_state=seed, stratify=y
        )

        # -------------------------
        # 4) Model + calibration
        # -------------------------
        use_xgb = bool(self.cfg.get("use_xgb", True))
        if use_xgb and _HAS_XGB:
            xgb_cfg = self.cfg.get("xgb", {}) or {}
            clf_base = xgb.XGBClassifier(
                n_estimators=int(xgb_cfg.get("n_estimators", 300)),
                max_depth=int(xgb_cfg.get("max_depth", 5)),
                learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
                subsample=float(xgb_cfg.get("subsample", 0.9)),
                colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.9)),
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )
        else:
            gbc_cfg = self.cfg.get("gbc", {}) or {}
            clf_base = GradientBoostingClassifier(
                n_estimators=int(gbc_cfg.get("n_estimators", 400)),
                learning_rate=float(gbc_cfg.get("learning_rate", 0.05)),
                max_depth=int(gbc_cfg.get("max_depth", 3)),
                subsample=float(gbc_cfg.get("subsample", 0.9)),
                random_state=seed,
            )

        # Isotonic calibration on a validation split
        clf = CalibratedClassifierCV(estimator=clf_base, method="isotonic", cv=3)
        clf.fit(X_train, y_train)
        p_val = clf.predict_proba(X_val)[:, 1]

        # -------------------------
        # 5) Per-domain thresholds from validation probs
        # -------------------------
        # Defaults if a domain is too small / degenerate
        default_low = float(self.cfg.get("thresholds", {}).get("default_low", 0.20))
        default_high = float(self.cfg.get("thresholds", {}).get("default_high", 0.60))

        domain_gates = {}
        for dd in np.unique(d_val):
            mask = (d_val == dd)
            if np.sum(mask) < 10:
                domain_gates[dd] = (default_low, default_high)
                continue
            lo, hi = _percentile_thresholds(p_val[mask], lo_pct=20.0, hi_pct=80.0)
            domain_gates[dd] = (lo, hi)

        # -------------------------
        # 6) Persist bundle (clf + feature names + version)
        # -------------------------
        out_dir = self._out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "clf": clf,
            "feature_names": base_feat_cols,  # order matters
            "version": "risk-bundle.v1",
        }
        bundle_path = out_dir / "bundle.joblib"
        dump(bundle, str(bundle_path))

        # Save thresholds
        gates_path = out_dir / "domain_thresholds.json"
        with gates_path.open("w", encoding="utf-8") as f:
            json.dump({k: {"low": v[0], "high": v[1]} for k, v in domain_gates.items()}, f, indent=2)

        # -------------------------
        # 7) Metrics
        # -------------------------
        auc = float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else float("nan")
        ap  = float(average_precision_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else float("nan")

        meta = {
            "n_total": int(len(df)),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "pos_rate_total": float(np.mean(y.values)),
            "pos_rate_val": float(np.mean(y_val)),
            "features": base_feat_cols,
            "auc_val": auc,
            "ap_val": ap,
            "thresholds": domain_gates,
            "bundle_path": str(bundle_path),
            "gates_path": str(gates_path),
        }
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

    def __init__(self, cfg: dict, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.run_dir: str = cfg.get("run_dir", "")
        self.data_path: str = cfg.get("data_path", "reports/risk_dataset.parquet")
        self.energy_thresh: float = float(cfg.get("energy_thresh", 0.80))
        self.cfg: dict = cfg.get("trainer", {}) or {}
        self.output_key: str = cfg.get("output_key", "risk_training")
        self.hot_reload_service: bool = bool(cfg.get("hot_reload_service", True))

        self.trainer = RiskTrainer(
            cfg=self.cfg, memory=memory, container=container, logger=logger
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
        # if self.hot_reload_service:
        #     try:
        #         svc = self.container.get("risk_predictor")
        #         if hasattr(svc, "reload_bundle") and callable(svc.reload_bundle):
        #             svc.reload_bundle()
        #             self.logger.log("RiskServiceReloaded", {"ok": True})
        #         else:
        #             self.logger.log("RiskServiceReloadSkipped", {"reason": "no reload_bundle()"})
        #     except Exception as e:
        #         self.logger.log("RiskServiceReloadFailed", {"error": str(e)})

        # 3) Return stats in context (mirrors your other agents)
        context[self.output_key] = {
            "status": "ok",
            "dataset_path": str(self.data_path),
            "model_out_dir": self.cfg.get("out_dir", "models/risk"),
            "meta": meta,
        }
        return context
