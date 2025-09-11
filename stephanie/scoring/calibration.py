# stephanie/scoring/calibration.py
import os
import json
import time
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

_logger = logging.getLogger(__name__)

class CalibrationManager:
    """Manages domain-aware calibration for knowledge retrieval."""
    
    def __init__(self, cfg: Dict, logger: Any = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.calibration_dir = Path(cfg.get("calibration_dir", "data/calibration"))
        self.history_dir = self.calibration_dir / "history"
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration parameters
        self.min_samples = cfg.get("min_calibration_samples", 100)
        self.retrain_days = cfg.get("calibration_retrain_days", 7)
        self.domain_hierarchy = cfg.get("domain_hierarchy", {})
        
        # Statistics tracking
        self.stats = {
            "calibration_requests": 0,
            "fallbacks": 0,
            "retrains": 0,
            "last_retrain": None
        }

    def get_confidence(self, domain: str, query: str, raw_similarity: float = 0.8) -> float:
        """
        Estimate confidence that calibration is reliable for this domain/query.
        Uses calibration curve + sample count.
        """
        cal = self.load_calibration(domain)
        coeffs = cal.get("ner", {}).get("coefficients")
        sample_count = cal.get("ner", {}).get("sample_count", 0)

        if not coeffs:
            return 0.5  # Neutral confidence if no calibration exists

        poly = np.poly1d(coeffs)
        calibrated = float(poly(raw_similarity))
        calibrated = max(0.0, min(1.0, calibrated))

        # Confidence is a mix of:
        # - sample size (trust calibration if lots of data)
        # - how close calibrated value is to [0,1] extremes
        size_factor = min(1.0, sample_count / self.min_samples)
        margin_factor = 1.0 - abs(0.5 - calibrated) * 2  # low if near 0.5, high if near 0/1

        return round(0.5 * size_factor + 0.5 * margin_factor, 3)

    def load_calibration(self, domain: str = "general") -> Dict:
        """Load calibration data with domain hierarchy fallbacks."""
        self.stats["calibration_requests"] += 1
        
        # Try specific domain
        cal = self._load_calibration_data(domain)
        if cal:
            return cal
            
        # Try parent domain
        parent_domain = self._get_parent_domain(domain)
        if parent_domain:
            cal = self._load_calibration_data(parent_domain)
            if cal:
                self._log_fallback(domain, parent_domain)
                return cal
                
        # Try general domain
        cal = self._load_calibration_data("general")
        if cal:
            self._log_fallback(domain, "general")
            return cal
            
        # Final fallback: identity function
        self._log_fallback(domain, "identity")
        return {
            "ner": {
                "coefficients": [1.0, 0.0],
                "description": "default_identity"
            }
        }

    def _load_calibration_data(self, domain: str) -> Optional[Dict]:
        """Load calibration data for a specific domain."""
        cal_path = self.calibration_dir / f"{domain}_calibration.json"
        if not cal_path.exists():
            return None
            
        try:
            with open(cal_path, "r") as f:
                cal_data = json.load(f)
                
            # Validate structure
            if "ner" not in cal_data or "coefficients" not in cal_data["ner"]:
                self.logger.warning(f"Invalid calibration format for domain: {domain}")
                return None
                
            return cal_data
            
        except Exception as e:
            self.logger.error(f"Failed to load calibration for {domain}: {e}")
            return None

    def _get_parent_domain(self, domain: str) -> Optional[str]:
        """Get parent domain from hierarchy configuration."""
        return self.domain_hierarchy.get(domain)

    def _log_fallback(self, requested: str, used: str):
        """Log domain fallback for monitoring."""
        self.stats["fallbacks"] += 1
        _logger.debug(f"Calibration fallback: {requested} → {used}")
        
        # Log to monitoring system
        if hasattr(self.logger, "log"):
            self.logger.log("CalibrationFallback", {
                "requested_domain": requested,
                "used_domain": used
            })

    def should_retrain(self, domain: str = "general") -> bool:
        """Determine if calibration should be retrained."""
        cal_path = self.calibration_dir / f"{domain}_calibration.json"
        if not cal_path.exists():
            return True
            
        try:
            # Check timestamp
            with open(cal_path, "r") as f:
                cal_data = json.load(f)
            last_train = datetime.fromisoformat(cal_data["timestamp"])
            if datetime.now() - last_train > timedelta(days=self.retrain_days):
                return True
                
            # Check data volume
            historical_data = self._load_historical_data(domain, days=1)
            return len(historical_data) >= self.min_samples
            
        except Exception as e:
            self.logger.error(f"Calibration timestamp check failed: {e}")
            return True

    def auto_train_calibration(self, domain: str = "general") -> bool:
        """Automatically train calibration if needed."""
        if not self.should_retrain(domain):
            return False
            
        historical_data = self._load_historical_data(domain)
        if not historical_data:
            return False
            
        try:
            # Train calibration
            calibration = self.train_calibration(historical_data, domain)
            
            # Save results
            self._save_calibration(domain, calibration, historical_data)
            
            # Log effectiveness
            self._log_calibration_effectiveness(domain, historical_data)
            
            self.stats["retrains"] += 1
            self.stats["last_retrain"] = datetime.now().isoformat()
            
            self.logger.info(f"Successfully retrained calibration for {domain}")
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration training failed for {domain}: {e}")
            return False

    def train_calibration(self, historical_data: List[Dict], domain: str) -> Dict:
        """Train calibration model from historical data."""
        # Filter valid data points
        valid_data = [
            d for d in historical_data 
            if "raw_similarity" in d and "is_relevant" in d
        ]
        
        if len(valid_data) < self.min_samples:
            raise ValueError(f"Not enough valid data for calibration: {len(valid_data)}/{self.min_samples}")
        
        # Extract features
        X = np.array([d["raw_similarity"] for d in valid_data])
        y = np.array([1.0 if d["is_relevant"] else 0.0 for d in valid_data])
        
        # Train polynomial model (degree 2 works well for this task)
        degree = self.cfg.get("calibration_degree", 2)
        coeffs = np.polyfit(X, y, degree)
        
        # Calculate model quality metrics
        y_pred = np.polyval(coeffs, X)
        mse = np.mean((y - y_pred) ** 2)
        
        return {
            "ner": {
                "coefficients": coeffs.tolist(),
                "degree": degree,
                "mse": float(mse),
                "sample_count": len(valid_data),
                "timestamp": datetime.now().isoformat()
            }
        }

    def _save_calibration(self, domain: str, calibration: Dict, historical_data: List[Dict]):
        """Save calibration data to disk."""
        # Save main calibration
        cal_path = self.calibration_dir / f"{domain}_calibration.json"
        with open(cal_path, "w") as f:
            json.dump(calibration, f, indent=2)
            
        # Archive historical data used for training
        archive_path = self.calibration_dir / "archive" / f"{domain}_{int(time.time())}_history.json"
        archive_path.parent.mkdir(exist_ok=True)
        with open(archive_path, "w") as f:
            json.dump(historical_data, f)

    def _load_historical_data(self, domain: str, days: int = 30) -> List[Dict]:
        """Load historical data for calibration within time window."""
        historical_data = []
        cutoff_time = datetime.now() - timedelta(days=days)
        
        for filename in os.listdir(self.history_dir):
            if not filename.endswith(".json"):
                continue
                
            filepath = self.history_dir / filename
            try:
                with open(filepath, "r") as f:
                    entry = json.load(f)
                    
                # Skip entries outside time window
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time < cutoff_time:
                    continue
                    
                # Filter by domain
                if domain == "all" or entry.get("domain") == domain:
                    historical_data.append(entry)
                    
            except Exception as e:
                self.logger.error(f"Calibration data load failed: {e}", extra={
                    "file": str(filepath)
                })
                
        return historical_data

    def log_calibration_event(self, 
                             query: str,
                             domain: str,
                             raw_similarity: float,
                             calibrated_similarity: float,
                             is_relevant: bool,
                             scorable_id: str,
                             scorable_type: str):
        """Log a calibration event for future training."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "domain": domain,
            "raw_similarity": raw_similarity,
            "calibrated_similarity": calibrated_similarity,
            "is_relevant": is_relevant,
            "scorable_id": scorable_id,
            "scorable_type": scorable_type
        }
        
        # Save to history
        filename = f"calibration_{int(time.time())}_{hash(json.dumps(event)) % 1000000}.json"
        with open(self.history_dir / filename, "w") as f:
            json.dump(event, f, indent=2)
            
        # Log for monitoring
        _logger.debug(f"Logged calibration event for '{query}' (domain: {domain})")

    def _log_calibration_effectiveness(self, domain: str, historical_data: List[Dict]):
        """Log metrics about how calibration affects retrieval quality."""
        if not historical_data:
            return
            
        # Calculate how calibration would change result distribution
        effects = []
        for entry in historical_data:
            if "raw_similarity" not in entry:
                continue
                
            # Simulate calibration
            cal = self.load_calibration(domain)
            if "ner" not in cal or "coefficients" not in cal["ner"]:
                continue
                
            poly = np.poly1d(cal["ner"]["coefficients"])
            calibrated_sim = float(poly(entry["raw_similarity"]))
            calibrated_sim = max(0.0, min(1.0, calibrated_sim))
            
            effects.append({
                "raw": entry["raw_similarity"],
                "calibrated": calibrated_sim,
                "delta": calibrated_sim - entry["raw_similarity"],
                "is_relevant": entry["is_relevant"]
            })
        
        if not effects:
            return
            
        # Calculate metrics
        deltas = [e["delta"] for e in effects]
        positive_shifts = [d for d in deltas if d > 0.05]
        negative_shifts = [d for d in deltas if d < -0.05]
        
        metrics = {
            "domain": domain,
            "total_effects": len(effects),
            "positive_shifts": len(positive_shifts),
            "negative_shifts": len(negative_shifts),
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "max_positive": float(max(deltas, default=0.0)),
            "max_negative": float(min(deltas, default=0.0)),
            "shift_ratio": len(positive_shifts) / max(1, len(effects)),
            "relevance_correlation": self._calculate_relevance_correlation(effects)
        }
        
        # Log to monitoring system
        self.logger.log("CalibrationEffectiveness", metrics)
        
        # Alert if calibration is causing excessive shifts
        if abs(metrics["mean_delta"]) > 0.2:
            self.logger.log("CalibrationWarning", {
                "message": "Calibration causing large mean shift",
                "domain": domain,
                "mean_delta": metrics["mean_delta"]
            })

    def _calculate_relevance_correlation(self, effects: List[Dict]) -> float:
        """Calculate how well calibration aligns with relevance."""
        relevant = [e for e in effects if e["is_relevant"]]
        non_relevant = [e for e in effects if not e["is_relevant"]]
        
        if not relevant or not non_relevant:
            return 0.0
            
        # Calculate mean calibrated similarity for relevant vs non-relevant
        relevant_mean = np.mean([e["calibrated"] for e in relevant])
        non_relevant_mean = np.mean([e["calibrated"] for e in non_relevant])
        
        # Simple correlation metric
        return max(0.0, min(1.0, relevant_mean - non_relevant_mean))

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get statistics about calibration system health."""
        return {
            **self.stats,
            "calibration_requests_per_hour": self._calculate_rate("calibration_requests"),
            "fallback_rate": self.stats["fallbacks"] / max(1, self.stats["calibration_requests"]),
            "retrain_interval_days": self.retrain_days,
            "min_samples": self.min_samples
        }

    def _calculate_rate(self, stat_name: str) -> float:
        """Calculate rate of a statistic per hour."""
        if not self.stats["last_retrain"] or self.stats[stat_name] == 0:
            return 0.0
            
        last_time = datetime.fromisoformat(self.stats["last_retrain"])
        hours = max(1.0, (datetime.now() - last_time).total_seconds() / 3600)
        return self.stats[stat_name] / hours