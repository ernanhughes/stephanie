# stephanie/scoring/calibration.py
import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

_logger = logging.getLogger(__name__)   

class CalibrationManager:
    """
    Learns P(relevant | raw_similarity, domain) using logistic/isotonic regression.
    Maintains one model per domain.
    """

    def __init__(self, cfg: dict, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.models: Dict[str, object] = {}  # domain -> model
        self.thresholds: Dict[str, float] = {}
        self.data_dir = cfg.get("data_dir", "data/calibration")
        os.makedirs(self.data_dir, exist_ok=True)
        self.min_samples = cfg.get("min_samples", 50)

    def log_event(self, domain: str, query: str, raw_sim: float, is_relevant: bool,
                  scorable_id: str, scorable_type: str, entity_type: str = None):
        """Log event to DB"""
        try:
            from stephanie.models.calibration import CalibrationEventORM
            event = CalibrationEventORM(
                domain=domain or "general",
                query=query,
                raw_similarity=raw_sim,
                is_relevant=is_relevant,
                scorable_id=scorable_id,
                scorable_type=scorable_type,
                entity_type=entity_type
            )
            self.memory.calibration_events.add(event)  # assuming memory has store
        except Exception as e:
            _logger.error(f"Failed to log calibration event: {e}")

    def load_data(self, domain: str) -> Optional[pd.DataFrame]:
        """Load all events for domain from DB"""
        try:
            rows = self.memory.calibration_events.get_by_domain(domain)
            if not rows:
                return None
            df = pd.DataFrame([r.to_dict() for r in rows])
            return df[["raw_similarity", "is_relevant"]]
        except Exception as e:
            _logger.error(f"Failed to load calibration data for {domain}: {e}")
            return None

    def train_model(self, domain: str) -> bool:
        df = self.load_data(domain)
        if df is None or len(df) < self.min_samples:
            return False

        X = df[["raw_similarity"]].values
        y = df["is_relevant"].values.astype(int)

        if len(set(y)) < 2:
            _logger.error(f"Skipping calibration for {domain}: only one class present")
            return False

        try:
            # Option 1: Isotonic Regression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(X.ravel(), y)

            # Option 2: Logistic Regression
            lr = LogisticRegression()
            lr.fit(X, y)

            # Choose best by F1
            iso_pred = (iso.predict(X.ravel()) > 0.5).astype(int)
            lr_pred = lr.predict(X)
            iso_f1 = f1_score(y, iso_pred)
            lr_f1 = f1_score(y, lr_pred)

            model = iso if iso_f1 >= lr_f1 else lr
            threshold = 0.5  # decision threshold on calibrated prob

            self.models[domain] = model
            self.thresholds[domain] = threshold

            _logger.info(f"Trained calibration model for {domain}: "
                           f"samples={len(df)}, use_iso={isinstance(model, IsotonicRegression)}")

            # Save model
            self._save_model(domain, model, threshold)
            return True

        except Exception as e:
            _logger.error(f"Training failed for domain {domain}: {e}", exc_info=True)
            return False

    def _save_model(self, domain: str, model, threshold: float):
        path = os.path.join(self.data_dir, f"{domain}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"model": model, "threshold": threshold}, f)

    def _load_model(self, domain: str):
        path = os.path.join(self.data_dir, f"{domain}.pkl")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.models[domain] = data["model"]
            self.thresholds[domain] = data["threshold"]
            return True
        except:
            return False

    def is_trained(self, domain: str) -> bool:
        return domain in self.models

    def get_calibrated_probability(self, domain: str, raw_sim: float) -> float:
        domain = domain or "general"
        if domain not in self.models:
            if not self._load_model(domain):
                return raw_sim  # fallback

        model = self.models[domain]
        sim = np.clip(raw_sim, 0, 1)
        if hasattr(model, "predict_proba"):
            return model.predict_proba([[sim]])[0, 1]
        else:
            return float(model.predict([sim]))

    def should_retrieve(self, domain: str, raw_sim: float, min_base: float = 0.45) -> bool:
        prob = self.get_calibrated_probability(domain, raw_sim)
        threshold = self.thresholds.get(domain, min_base)
        return prob >= threshold

    def get_confidence(self, domain: str, query: str = None) -> float:
        """How confident we are in our calibration for this domain"""
        if not self.is_trained(domain):
            return 0.0
        count = self.memory.calibration_events.count_by_domain(domain)
        return min(count / 1000, 1.0)  # cap at 1.0 after 1k samples