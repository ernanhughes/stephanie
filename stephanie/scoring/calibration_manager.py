# stephanie/scoring/calibration_manager.py
import glob
import logging
import os
import pickle
import re
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
        self.logger = logger
        self.models: Dict[str, object] = {}      # hydrated calibrators (domain -> model)
        self.thresholds: Dict[str, float] = {}   # domain -> decision threshold
        self.min_samples = int(self.cfg.get("min_samples", 50))
        self.data_dir = str(self.cfg.get("data_dir", "/data/calibration"))

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

    def domain_counts(self, domain: str) -> Tuple[int, int, int]:
        """(pos, neg, total) counts from the store."""
        try:
            counts = self.memory.calibration_events.count_by_domain(domain) or {}
            pos = int(counts.get("pos", 0))
            neg = int(counts.get("neg", 0))
            return pos, neg, pos + neg
        except Exception as e:
            _logger.error(f"domain_counts failed for {domain}: {e}")
            return 0, 0, 0

    def _load_training(self, domain: str):
        """
        Return (X, y, raw) where:
        - X: 2D feature matrix (coverage, correctness, coherence, citation_support)
        - y: binary labels (1=relevant, 0=not)
        - raw: raw similarity scores (float in [0,1])
        """
        events = self.memory.calibration_events.fetch_events(domain, limit=10000) or []
        X, y, raw = [], [], []
        for ev in events:
            f = (ev.get("features") or {})
            X.append([
                float(f.get("coverage", 0.0)),
                float(f.get("correctness", 0.0)),
                float(f.get("coherence", 0.0)),
                float(f.get("citation_support", 0.0)),
            ])
            y.append(1 if ev.get("is_relevant") else 0)
            raw.append(float(ev.get("raw_similarity", 0.0)))
        return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(raw, dtype=float)

    # ----------------------------
    # Training
    # ----------------------------
    def train_model(self, domain: str, allow_fallback: bool = True) -> bool:
        """
        Train a per-domain calibrator.
        - Normal path: logistic regression on feature matrix X -> P(relevant)
        - One-class path: quantile threshold fallback on raw similarity (or mean(X))
        - No-sklearn path: parametric sigmoid on raw similarity
        """
        X, y, raw = self._load_training(domain)  # <-- THREE returns
        n = int(X.shape[0])
        if n == 0:
            return False

        uniq = np.unique(y)
        # ----- One-class / no-class fallback -----
        if uniq.size < 2:
            if not allow_fallback:
                return False

            positive = bool(uniq[0]) if uniq.size == 1 else True
            q = float(self.cfg.get("calibration", {}).get("fallback_quantile", 0.85))
            slope = float(self.cfg.get("calibration", {}).get("fallback_slope", 10.0))

            # Use raw similarity if available, otherwise mean of features
            source = raw if raw.size else (X.mean(axis=1) if X.size else np.zeros((n,), dtype=float))

            cal = QuantileThresholdCalibrator(positive=positive, quantile=q, slope=slope)
            cal.fit(source)

            # Persist with a sensible threshold (the learned quantile)
            thresh = float(np.quantile(source, q)) if source.size else 0.5
            self._persist(domain, cal, kind="quantile", threshold=thresh)
            self.logger.info(
                "Calibration: trained fallback quantile model",
                extra={"domain": domain, "positive": positive, "n": n, "quantile": q, "threshold": thresh},
            )
            return True

        # ----- Two-class path -----
        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(max_iter=1000)

            # If X has no columns or is degenerate, regress on raw similarity instead
            degenerate = (X.ndim != 2) or (X.shape[1] == 0) or np.allclose(X.std(axis=0), 0.0)
            feats = raw.reshape(-1, 1) if (degenerate and raw.size) else X

            clf.fit(feats, y)
            # Use 0.5 as default decision threshold for P(relevant)
            self._persist(domain, clf, kind="logistic", threshold=0.5)
            self.logger.info("Calibration: trained logistic", extra={"domain": domain, "n": n, "degenerate_X": bool(degenerate)})
            return True

        except Exception as e:
            # ----- No-sklearn (or import) fallback: simple sigmoid on raw similarity -----
            mu = float(raw.mean()) if raw.size else 0.0
            sd = float(raw.std() or 1.0) if raw.size else 1.0

            class _Sigmoid:
                def __init__(self, mu, sd):
                    self.mu, self.sd = mu, sd
                def predict_proba(self, scores):
                    z = (np.asarray(scores, float) - self.mu) / self.sd
                    p = 1.0 / (1.0 + np.exp(-z))
                    return np.stack([1 - p, p], axis=1)

            cal = _Sigmoid(mu, sd)
            self._persist(domain, cal, kind="sigmoid_mu_sd", threshold=0.5)
            self.logger.info(
                "Calibration: trained simple sigmoid (fallback)",
                extra={"domain": domain, "n": n, "mu": mu, "sd": sd, "reason": str(e)},
            )
            return True

    def _slug(self, name: str) -> str:
        # safe filename for domain names like "ml/nlp:general"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "general"))

    def _cal_path(self, domain: str, kind: str | None = None) -> str:
        # Prefer namespaced files: <domain>__<kind>.pkl; fallback to <domain>.pkl
        slug = self._slug(domain)
        if kind:
            return os.path.join(self.data_dir, f"{slug}__{kind}.pkl")
        return os.path.join(self.data_dir, f"{slug}.pkl")

    def _dump_pickle_atomic(self, path: str, obj: object) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)  # atomic on POSIX/NTFS

    def _load_pickle_lenient(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Support legacy shapes:
        # 1) {"model": <clf>, "threshold": 0.5}
        # 2) {"calibrator": <clf>, "meta": {"threshold": 0.5, "kind": "..."}}
        # 3) (<clf>, {"threshold": 0.5, "kind": "..."})
        if isinstance(data, dict):
            if "calibrator" in data:
                return data["calibrator"], data.get("meta", {})
            if "model" in data:
                return data["model"], {"threshold": float(data.get("threshold", 0.5))}
        if isinstance(data, tuple) and len(data) >= 2:
            return data[0], (data[1] or {})
        # As a last resort treat the whole payload as the calibrator
        return data, {}

    # ----------------------------
    # Persist
    # ----------------------------
    def _persist(self, domain: str, calibrator: object, kind: str, threshold: float = 0.5) -> None:
        """Persist calibrator to local disk (atomic pickle) and hydrate cache."""
        try:
            path = self._cal_path(domain, kind)
            payload = {
                "calibrator": calibrator,
                "meta": {"threshold": float(threshold), "kind": kind, "version": 1},
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Save namespaced file
            self._dump_pickle_atomic(path, payload)

            # Also save/refresh a domain-wide pointer for quick lookup (optional)
            # This allows _ensure_loaded(domain) to work without knowing 'kind'
            generic_path = self._cal_path(domain)
            self._dump_pickle_atomic(generic_path, payload)

        except Exception as e:
            # Local persistence should never crash the pipeline
            self.logger.error(f"calibration._persist failed for {domain}: {e}")

        # Hydrate local cache so immediate calls work
        self.models[domain] = calibrator
        self.thresholds[domain] = float(threshold)

    # ----------------------------
    # Inference
    # ----------------------------
    def _ensure_loaded(self, domain: str) -> bool:
        """Load calibrator from local files into memory if needed."""
        if domain in self.models:
            return True

        # Try generic domain file first
        try:
            generic_path = self._cal_path(domain)
            if os.path.exists(generic_path):
                cal, meta = self._load_pickle_lenient(generic_path)
                thr = float((meta or {}).get("threshold", 0.5))
                self.models[domain] = cal
                self.thresholds[domain] = thr
                return True
        except Exception as e:
            self.logger.error(f"calibration.load (generic) failed for {domain}: {e}")

        # Then try any namespaced file <domain>__*.pkl
        try:
            pattern = self._cal_path(domain, kind="*")
            for path in glob.glob(pattern):
                cal, meta = self._load_pickle_lenient(path)
                thr = float((meta or {}).get("threshold", 0.5))
                self.models[domain] = cal
                self.thresholds[domain] = thr
                return True
        except Exception as e:
            self.logger.error(f"calibration.load (namespaced) failed for {domain}: {e}")

        # (Optional) final fallback: try DB store if you keep it around
        try:
            if hasattr(self.memory, "calibration_events") and hasattr(self.memory.calibration_events, "load_calibrator"):
                loaded = self.memory.calibration_events.load_calibrator(domain)
                if loaded:
                    if isinstance(loaded, tuple) and len(loaded) >= 2:
                        cal, meta = loaded[0], loaded[1]
                        thr = float((meta or {}).get("threshold", 0.5))
                    elif isinstance(loaded, dict):
                        cal = loaded.get("calibrator") or loaded.get("model")
                        thr = float(loaded.get("threshold", 0.5))
                    else:
                        cal, thr = loaded, 0.5
                    if cal is not None:
                        self.models[domain] = cal
                        self.thresholds[domain] = thr
                        # Re-persist locally for next time
                        self._persist(domain, cal, kind=(meta or {}).get("kind", "restored"), threshold=thr)
                        return True
        except Exception as e:
            self.logger.error(f"calibration.load (db fallback) failed for {domain}: {e}")

        return False

    def is_trained(self, domain: str) -> bool:
        return domain in self.models or os.path.exists(self._cal_path(domain))

    def _ensure_loaded(self, domain: str) -> bool:
        if domain in self.models:
            return True
        # tolerate stores without the new API
        store = getattr(self.memory, "calibration_events", None)
        loader = getattr(store, "load_calibrator", None)
        data = loader(domain) if callable(loader) else None
        if not data:
            # sane identity default
            data = {"coefficients": [1.0, 0.0]}
            if self.logger:
                self.logger.log("CalibrationDefaultUsed", {"domain": domain})
        self.models[domain] = data
        return True

    def get_calibrated_probability(self, domain: str, raw_sim: float) -> float:
        self._ensure_loaded(domain)
        model = self.models.get(domain) or {"coefficients": [1.0, 0.0]}
        coeffs = model.get("coefficients", [1.0, 0.0])
        # polynomial transform; clamp to [0,1]
        try:
            import numpy as np
            val = float(np.polyval(coeffs, raw_sim))
        except Exception:
            val = float(raw_sim)
        return max(0.0, min(1.0, val))

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
        """
        Return [0,1] expressing how confident we are in the calibration for `domain`.
        Works whether the store returns an int or a dict of counts.
        Normalizes by a configurable cap (default 1000 labeled examples).
        """
        cap = float(self.cfg.get("calibration", {}).get("confidence_cap", 1000.0))
        total = 0
        try:
            counts = self.memory.calibration_events.count_by_domain(domain)
            if isinstance(counts, dict):
                total = int(counts.get("total") or (counts.get("pos", 0) + counts.get("neg", 0)))
            else:
                total = int(counts or 0)
        except Exception as e:
            _logger.warning(f"get_confidence: count fetch failed for domain={domain}: {e}")
            total = 0

        # Avoid div-by-zero and clamp to [0,1]
        if cap <= 0:
            return 0.0
        return float(min(total / cap, 1.0))
