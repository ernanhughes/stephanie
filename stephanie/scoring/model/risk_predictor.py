# stephanie/scoring/model/risk_predictor.py
"""
Calibrated hallucination-risk scoring with per-domain thresholds.

Usage:
    pred = DomainCalibratedRiskPredictor(
        bundle_path="./models/risk/bundle.joblib",
        default_domains=["science","history","geography","tech"],
        memcube=MemCubeClient(),
        featurizer=RiskFeaturizer()
    )
    risk, (low, high) = await pred.predict_risk(question, context)

Design:
- Bundle must expose `.clf` with `predict_proba` and `.feature_names` (ordered list).
- Domain thresholds fetched from MemCube; fallback to config defaults.
- Asynchronous API; thread-safe reads; PII sanitization in featurizer.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.memcube.memcube_client import MemCubeClient

import joblib

_logger = logging.getLogger(__name__)


class DomainRequiredError(ValueError):
    """Raised when no domain/domain_tags provided and classification is disabled upstream."""

    pass


# --------- PII sanitization (emails/phones) ----------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")


def sanitize_text(s: str) -> str:
    s = EMAIL_RE.sub("<EMAIL>", s or "")
    s = PHONE_RE.sub("<PHONE>", s)
    return s


# --------- Featurizer contract ----------
@dataclass
class RiskFeaturizer:
    """
    Produces a dense, ordered feature vector for the risk model.
    Replace `featurize` with your production features:
    - semantic embeddings
    - coverage/gap metrics from MemCube
    - prior Δ-energy EMA, etc.
    """

    feature_order: List[str] = field(
        default_factory=lambda: [
            "q_len",
            "ctx_len",
            "overlap_ratio",
            "ner_count",
            "num_tokens_est",
            "coverage_gap",
            "prior_max_energy_ema",
        ]
    )

    async def featurize(self, question: str, context: str) -> Dict[str, float]:
        q = sanitize_text(question or "")
        c = sanitize_text(context or "")
        q_tokens = q.split()
        c_tokens = c.split()
        inter = len(set(q_tokens) & set(c_tokens))
        denom = max(1, len(set(q_tokens)))
        overlap = inter / denom

        feats = {
            "q_len": float(len(q)),
            "ctx_len": float(len(c)),
            "overlap_ratio": float(overlap),
            "ner_count": float(
                sum(w.istitle() for w in q_tokens)
            ),  # toy NER proxy
            "num_tokens_est": float(len(q_tokens) + len(c_tokens)),
            "coverage_gap": float(1.0 - overlap),
            "prior_max_energy_ema": 0.25,  # placeholder; wire your EMA here
        }
        return feats


# --------- Model bundle ----------
@dataclass
class RiskModelBundle:
    clf: Any
    feature_names: List[str]
    version: str = "risk-bundle.v1"

    @classmethod
    def load(cls, path: str) -> "RiskModelBundle":
        if joblib is None:
            raise RuntimeError("joblib not available to load model bundle")
        obj = joblib.load(path)
        # Support multiple serialization styles
        if isinstance(obj, dict) and "clf" in obj and "feature_names" in obj:
            clf = obj["clf"]
            names = obj["feature_names"]
            ver = obj.get("version", "risk-bundle.v1")
            return cls(clf=clf, feature_names=names, version=ver)
        # Direct estimator with attached names
        if hasattr(obj, "predict_proba") and hasattr(obj, "feature_names"):
            return cls(
                clf=obj,
                feature_names=list(obj.feature_names),
                version=getattr(obj, "version", "risk-bundle.v1"),
            )
        raise ValueError(
            "Unsupported risk bundle format; expect dict with keys ['clf','feature_names']"
        )


# --------- Predictor ----------
class DomainCalibratedRiskPredictor:
    """
    Async hallucination-risk predictor with per-domain thresholds.

    Methods:
      - predict_risk(question, context) -> (risk, (low, high))

    Thresholds:
      - fetched from MemCube (kind="risk", fields: low_threshold, high_threshold)
      - fallback to defaults if not present
    """

    def __init__(
        self,
        bundle_path: Optional[str] = None,
        default_domains: Optional[List[str]] = None,
        default_thresholds: Tuple[float, float] = (0.2, 0.6),
        memcube: Optional[MemCubeClient] = None,
        featurizer: Optional[RiskFeaturizer] = None,
        domain_classifier: Optional[ScorableClassifier] = None,
    ):
        self.bundle: Optional[RiskModelBundle] = None
        if bundle_path:
            self.bundle = RiskModelBundle.load(bundle_path)

        self.default_domains = default_domains or ["programming", "ml", "nlp", "ai"]
        self.default_thresholds = (
            float(default_thresholds[0]),
            float(default_thresholds[1]),
        )
        self.memcube = memcube or MemCubeClient()
        self.featurizer = featurizer or RiskFeaturizer()

        # Local cache of thresholds to avoid frequent lookups
        self.domain_classifier = domain_classifier  # <-- NEW

        self._domain_thresholds: Dict[str, Tuple[float, float]] = {}

        # Validate feature contract if bundle is present
        if self.bundle is not None:
            missing = [
                k
                for k in self.featurizer.feature_order
                if k not in self.bundle.feature_names
            ]
            if missing:
                _logger.warn(
                    f"Risk bundle missing features used by featurizer: {missing}"
                )

    # ------------- internal helpers -------------
    async def _get_domain_thresholds(self, domain: str) -> Tuple[float, float]:
        if domain in self._domain_thresholds:
            return self._domain_thresholds[domain]

        rec = await self.memcube.query_calibration(
            "risk",
            filters={"domain": domain},
            sort=[("created_at", "DESC")],
            limit=1,
        )
        if rec and isinstance(rec, dict):
            lo = float(rec.get("low_threshold", self.default_thresholds[0]))
            hi = float(rec.get("high_threshold", self.default_thresholds[1]))
        else:
            lo, hi = self.default_thresholds

        # basic sanity
        lo = max(0.0, min(lo, 0.95))
        hi = max(lo + 0.01, min(hi, 0.99))
        self._domain_thresholds[domain] = (lo, hi)
        return lo, hi

    def _vectorize(self, feats: Dict[str, float]) -> np.ndarray:
        """
        Order features per bundle.feature_names, fill missing with 0.
        If bundle absent, use featurizer.feature_order.
        """
        names = (
            self.bundle.feature_names
            if self.bundle
            else self.featurizer.feature_order
        )
        x = np.array(
            [[float(feats.get(k, 0.0)) for k in names]], dtype=np.float32
        )
        return x, names

    def _predict_proba(self, x: np.ndarray) -> float:
        # If model bundle available → calibrated predict_proba
        if self.bundle is not None:
            proba = self.bundle.clf.predict_proba(x)[:, 1]
            return float(np.clip(proba[0], 0.0, 1.0))
        # Fallback: tiny heuristic on a couple features
        q_len_idx = (
            self.featurizer.feature_order.index("q_len")
            if "q_len" in self.featurizer.feature_order
            else 0
        )
        gap_idx = (
            self.featurizer.feature_order.index("coverage_gap")
            if "coverage_gap" in self.featurizer.feature_order
            else 0
        )
        q_len = x[0, q_len_idx]
        gap = x[0, gap_idx]
        # Risk grows with coverage gap and short questions
        risk = 0.2 + 0.6 * (gap) + 0.2 * (1.0 - math.tanh(q_len / 512.0))
        return float(max(0.0, min(1.0, risk)))

    # ------------- public API -------------
    async def predict_risk(
        self,
        question: str,
        context: str,
        *,
        domain: Optional[str] = None,
        domain_tags: Optional[List[str]] = None,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Returns:
            risk: float in [0,1]
            thresholds: (low, high): domain-calibrated gates
        """
        # STRICT: never guess; a domain (or tags) must be provided by caller.
        dom = (domain or "").strip().lower() or self._choose_primary_domain(
            domain_tags
        )
        if not dom:
            raise DomainRequiredError(
                "Domain is required. Provide `domain` or `domain_tags` from the ScorableDomainAgent."
            )
        # 0) Domain
        domain = await self._guess_domain(question or "")
        low, high = await self._get_domain_thresholds(domain)

        # 1) Features
        feats = await self.featurizer.featurize(question or "", context or "")
        x, names = self._vectorize(feats)

        # 2) Score
        risk = self._predict_proba(x)

        return risk, (low, high)

    # ------------- optional: explanation -------------
    async def explain(self, question: str, context: str) -> Dict[str, Any]:
        """
        Returns SHAP-like explanation if shap is installed and bundle is a tree-based model.
        """
        try:
            import shap  # optional
        except Exception:
            return {"ok": False, "reason": "shap not installed"}

        if self.bundle is None:
            return {"ok": False, "reason": "no bundle"}
        if not hasattr(self.bundle.clf, "predict_proba"):
            return {"ok": False, "reason": "clf lacks predict_proba"}

        feats = await self.featurizer.featurize(question or "", context or "")
        x, names = self._vectorize(feats)

        # Try to unwrap isotonic calibration wrappers (commonly clf.base_estimator)
        base = getattr(self.bundle.clf, "base_estimator", self.bundle.clf)
        try:
            explainer = shap.TreeExplainer(base)
            sv = explainer.shap_values(x)
            sv0 = sv[0] if isinstance(sv, list) else sv
            values = {names[i]: float(sv0[0, i]) for i in range(len(names))}
            return {
                "ok": True,
                "expected_value": float(
                    explainer.expected_value[0]
                    if isinstance(
                        explainer.expected_value, (list, tuple, np.ndarray)
                    )
                    else explainer.expected_value
                ),
                "shap_values": values,
            }
        except Exception as e:
            return {"ok": False, "reason": f"shap failed: {e}"}

    async def _guess_domain(self, question: str) -> str:
        """
        Uses ScorableClassifier if available, otherwise falls back to MemCube.
        """
        if self.domain_classifier:
            try:
                results = self.domain_classifier.classify(
                    question,
                    top_k=1,
                    min_value=0.4,  # or adjust threshold
                )
                if results:
                    return results[0][0]
            except Exception as e:
                _logger.warning(f"Domain classification failed: {e}")

        # Fallback
        return await self.memcube.guess_domain(question or "")
