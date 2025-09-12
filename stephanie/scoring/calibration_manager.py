# stephanie/scoring/calibration_manager.py
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from stephanie.models.quantile_fallback import QuantileThresholdCalibrator

_logger = logging.getLogger(__name__)


class SigmoidCalibrator:
    """
    Simple, sklearn-free calibrator on 1D raw scores.
    p = sigmoid((x - mu) / sd)
    """
    def __init__(self, mu: float, sd: float):
        self.mu = float(mu)
        self.sd = float(sd if sd and sd > 0 else 1.0)

    def predict_proba(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        z = (s - self.mu) / self.sd
        p = 1.0 / (1.0 + np.exp(-z))
        return np.hstack([1.0 - p, p])

    def predict(self, scores) -> np.ndarray:
        return self.predict_proba(scores)[:, 1]


class CalibrationManager:
    """
    Learns P(relevant | raw_similarity, domain).
    - Persists calibrators via memory.calibration_events.{persist,load}_calibrator
    - Trains on raw_similarity only (keeps train/infer consistent)
    - Handles one-class fallback via QuantileThresholdCalibrator
    - Handles two-class without sklearn via SigmoidCalibrator
    """

    def __init__(self, cfg: dict, memory, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger or _logger
        self.models: Dict[str, object] = {}      # hydrated calibrators (domain -> model)
        self.thresholds: Dict[str, float] = {}   # domain -> decision threshold
        self.min_samples = int(self.cfg.get("min_samples", 50))

    # ----------------------------
    # Logging API
    # ----------------------------
    def log_event(
        self,
        domain: str,
        query: str,
        raw_sim: float,
        is_relevant: bool,
        scorable_id: str,
        scorable_type: str,
        entity_type: str = None,
    ):
        """Log event to DB (NumPy-safe)."""
        try:
            from stephanie.models.calibration import CalibrationEventORM

            def _f(x, default=0.0) -> float:
                try:
                    return float(x)
                except Exception:
                    return float(default)

            def _b(x) -> bool:
                try:
                    return bool(x)
                except Exception:
                    return False

            def _s(x) -> str:
                try:
                    return x if isinstance(x, str) else str(x)
                except Exception:
                    return ""

            event = CalibrationEventORM(
                domain=_s(domain) or "general",
                query=_s(query)[:2000],
                raw_similarity=_f(raw_sim, 0.0),
                is_relevant=_b(is_relevant),
                scorable_id=_s(scorable_id),
                scorable_type=_s(scorable_type),
                entity_type=_s(entity_type) if entity_type is not None else None,
            )
            self.memory.calibration_events.add(event)
        except Exception as e:
            _logger.error(f"Failed to log calibration event: {e}")

    # ----------------------------
    # Data access helpers
    # ----------------------------
    def load_data(self, domain: str) -> Optional[pd.DataFrame]:
        """Load all events for domain from DB (raw_similarity + is_relevant)."""
        try:
            rows = self.memory.calibration_events.get_by_domain(domain)
            if not rows:
                return None
            df = pd.DataFrame([r.to_dict() for r in rows])
            if "raw_similarity" not in df or "is_relevant" not in df:
                return None
            return df[["raw_similarity", "is_relevant"]].dropna()
        except Exception as e:
            _logger.error(f"Failed to load calibration data for {domain}: {e}")
            return None

    def label_counts(self, domain: str) -> Tuple[int, int, int]:
        """(pos, neg, total) counts from the store."""
        try:
            counts = self.memory.calibration_events.fetch_counts_by_label(domain) or {}
            pos = int(counts.get("pos", 0))
            neg = int(counts.get("neg", 0))
            return pos, neg, pos + neg
        except Exception as e:
            _logger.error(f"label_counts failed for {domain}: {e}")
            return 0, 0, 0

    def _load_training(self, domain: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns arrays for training on 1D raw_similarity.
        X is shape (N, 1), y is (N,)
        """
        try:
            events = self.memory.calibration_events.fetch_events(domain, limit=10000) or []
        except Exception:
            # Fallback: derive from ORM → dicts
            df = self.load_data(domain)
            if df is None:
                return np.zeros((0, 1), float), np.zeros((0,), int)
            X = df["raw_similarity"].to_numpy(dtype=float).reshape(-1, 1)
            y = df["is_relevant"].astype(int).to_numpy()
            return X, y

        rs, y = [], []
        for ev in events:
            # ev is dict-like
            rs.append(float(ev.get("raw_similarity", 0.0)))
            y.append(1 if ev.get("is_relevant") else 0)
        X = np.asarray(rs, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        return X, y

    # ----------------------------
    # Training
    # ----------------------------
    def train_model(self, domain: str, allow_fallback: bool = True) -> bool:
        X, y = self._load_training(domain)
        n = X.shape[0]
        if n < max(5, self.min_samples // 2):  # light guard for early stages
            self.logger.info("Calibration: not enough samples", extra={"domain": domain, "n": int(n)})
            return False

        uniq = np.unique(y)
        # ---- One-class fallback ----
        if uniq.size < 2:
            if not allow_fallback:
                return False

            positive = bool(uniq[0]) if uniq.size == 1 else True
            q = float(self.cfg.get("calibration", {}).get("fallback_quantile", 0.85))
            slope = float(self.cfg.get("calibration", {}).get("fallback_slope", 10.0))

            cal = QuantileThresholdCalibrator(
                positive=positive,
                quantile=q,
                slope=slope,
            )
            cal.fit(X.ravel())  # expects 1D
            threshold = 0.5  # probability threshold
            self._persist(domain, cal, kind="quantile", threshold=threshold)
            self.logger.info("Calibration: trained fallback quantile model",
                             extra={"domain": domain, "positive": positive, "n": int(n), "q": q})
            return True

        # ---- Two-class path ----
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X, y)
            threshold = 0.5
            self._persist(domain, clf, kind="logistic_raw", threshold=threshold)
            self.logger.info("Calibration: trained logistic on raw", extra={"domain": domain, "n": int(n)})
            return True
        except Exception:
            # sklearn not available → simple sigmoid calibrated on raw
            pos = X[y == 1].ravel()
            neg = X[y == 0].ravel()
            mu_pos = float(pos.mean()) if pos.size else 0.75
            mu_neg = float(neg.mean()) if neg.size else 0.25
            mu = (mu_pos + mu_neg) / 2.0
            sd = float(np.sqrt(((pos.std() ** 2) + (neg.std() ** 2)) / 2.0) or 1.0)
            cal = SigmoidCalibrator(mu=mu, sd=sd)
            threshold = 0.5
            self._persist(domain, cal, kind="sigmoid_raw", threshold=threshold)
            self.logger.info("Calibration: trained sigmoid fallback",
                             extra={"domain": domain, "mu": round(mu, 4), "sd": round(sd, 4)})
            return True

    def _persist(self, domain: str, calibrator: object, kind: str, threshold: float = 0.5) -> None:
        """Persist via the calibration_events store and hydrate local cache."""
        try:
            self.memory.calibration_events.persist_calibrator(domain, calibrator, kind=kind, threshold=threshold)
        except Exception as e:
            self.logger.error(f"persist_calibrator failed for {domain}: {e}")
        # hydrate local cache so immediate calls work
        self.models[domain] = calibrator
        self.thresholds[domain] = float(threshold)

    # ----------------------------
    # Inference
    # ----------------------------
    def _ensure_loaded(self, domain: str) -> bool:
        """Load calibrator from store into memory if needed."""
        if domain in self.models:
            return True
        try:
            loaded = self.memory.calibration_events.load_calibrator(domain)
        except Exception as e:
            self.logger.error(f"load_calibrator failed for {domain}: {e}")
            loaded = None

        if not loaded:
            return False

        # support either tuple or dict from store
        if isinstance(loaded, tuple) and len(loaded) >= 2:
            calibrator, meta = loaded[0], loaded[1]
            threshold = float((meta or {}).get("threshold", 0.5))
        elif isinstance(loaded, dict):
            calibrator = loaded.get("calibrator") or loaded.get("model")
            threshold = float(loaded.get("threshold", 0.5))
        else:
            calibrator, threshold = loaded, 0.5

        if calibrator is None:
            return False

        self.models[domain] = calibrator
        self.thresholds[domain] = threshold
        return True

    def is_trained(self, domain: str) -> bool:
        if domain in self.models:
            return True
        return self._ensure_loaded(domain)

    def get_calibrated_probability(self, domain: str, raw_sim: float) -> float:
        domain = domain or "general"
        if not self._ensure_loaded(domain):
            # no calibrator available → return raw
            return float(np.clip(raw_sim, 0.0, 1.0))

        model = self.models[domain]
        sim = float(np.clip(raw_sim, 0.0, 1.0))
        # Try predict_proba first
        if hasattr(model, "predict_proba"):
            try:
                return float(model.predict_proba([[sim]])[0, 1])
            except Exception:
                # some custom calibrators accept 1D
                return float(model.predict_proba([sim])[0, 1])
        # Fallback to predict (returning prob of positive class)
        if hasattr(model, "predict"):
            try:
                out = model.predict([[sim]])
            except Exception:
                out = model.predict([sim])
            # may return (N,) or (N,1)
            out = np.asarray(out, dtype=float).ravel()
            return float(out[0])
        # last resort
        return sim

    def should_retrieve(self, domain: str, raw_sim: float, min_base: float = 0.45) -> bool:
        prob = self.get_calibrated_probability(domain, raw_sim)
        threshold = float(self.thresholds.get(domain, min_base))
        return prob >= threshold

    def get_confidence(self, domain: str, query: str = None) -> float:
        """How confident we are in our calibration for this domain (by sample count)."""
        try:
            count = self.memory.calibration_events.count_by_domain(domain)
        except Exception:
            count = 0
        # cap at 1.0 after 1k samples
        return float(min(count / 1000.0, 1.0))
