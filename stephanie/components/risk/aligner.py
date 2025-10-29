# stephanie/components/gap/risk/aligner.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Optional import: if GAP shared_scm is present, we expose to_scm(); otherwise it's a no-op.
try:
    from stephanie.core.shared_scm import scm_row  # type: ignore
except Exception:  # pragma: no cover - optional
    def scm_row(**kwargs):  # type: ignore
        return kwargs


# ------------------------------ helpers -------------------------------------

def _clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_get(d: Dict[str, Any], k: str, default: float = 0.5) -> float:
    try:
        return float(d.get(k, default))
    except Exception:
        return float(default)


def _invert01(x: float) -> float:
    return _clamp01(1.0 - float(x))


# ------------------------------ calibration ---------------------------------

@dataclass
class MonotoneCalibrator:
    """
    Minimal monotone calibrator: 'affine' or 'logistic'.
    Example params:
      - affine:  {'a': 1.0, 'b': 0.0}
      - logistic:{'k': 6.0, 'x0': 0.5}
    """
    kind: str = "logistic"
    params: Optional[Dict[str, float]] = None

    def __call__(self, value: float) -> float:
        x = float(value)
        if self.kind == "affine":
            a = (self.params or {}).get("a", 1.0)
            b = (self.params or {}).get("b", 0.0)
            return _clamp01(a * x + b)
        # logistic
        p = self.params or {}
        k = p.get("k", 6.0)
        x0 = p.get("x0", 0.5)
        try:
            import math
            y = 1.0 / (1.0 + math.exp(-k * (x - x0)))
        except OverflowError:
            y = 0.0 if (x - x0) < 0 else 1.0
        return _clamp01(y)


# ------------------------------ Aligner -------------------------------------

class Aligner:
    """
    Normalizes raw monitor metrics to [0,1], applies optional per-metric calibration,
    and computes Δ-gap when a baseline/reference is provided in `context`.

    Inputs (raw metrics; names are flexible, we adapt):
      - confidence01 OR (uncertainty01 -> invert)
      - faithfulness_risk01 OR halluc_risk01 OR (faithfulness01 -> invert)
      - ood_hat01 OR ood_risk01 OR (in_domain01 -> invert)
      - delta_gap01 (optional direct; otherwise computed/estimated)

    Context (all optional):
      - context['baseline_metrics'] or context['reference_metrics']: Dict[str, float] in [0,1]
      - context['disagreement01']: precomputed Δ in [0,1]
      - context['evidence']: { 'coverage01': ..., 'consistency01' or 'faithfulness01': ... }

    Public API:
      normalize(raw_metrics, context=None) -> Dict[str, float]  # keys: confidence01, faithfulness_risk01, ood_hat01, delta_gap01
      to_scm(goal, reply, metrics01) -> Dict[str, Any]          # convenience for persistence
    """

    def __init__(
        self,
        *,
        calibration_params: Optional[Dict[str, Dict[str, float]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        cal = calibration_params or {}

        # Per-metric calibrators (use logistic by default)
        self.c_conf = MonotoneCalibrator("logistic", cal.get("confidence01"))
        self.c_faith = MonotoneCalibrator("logistic", cal.get("faithfulness_risk01"))
        self.c_ood = MonotoneCalibrator("logistic", cal.get("ood_hat01"))
        self.c_delta = MonotoneCalibrator("logistic", cal.get("delta_gap01"))

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    def normalize(self, raw: Dict[str, Any], *, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Returns normalized metrics dict in [0,1]. Robust to missing fields.
        If a baseline is present in context, computes Δ-gap against baseline.
        """
        # ---- confidence ----
        conf = _safe_get(raw, "confidence01", None)  # type: ignore[arg-type]
        if conf is None:
            conf = _invert01(_safe_get(raw, "uncertainty01", 0.5))
        conf = self.c_conf(_clamp01(conf))

        # ---- faithfulness risk ----
        faith_risk = raw.get("faithfulness_risk01", None)
        if faith_risk is None:
            # Fallbacks: halluc_risk01, or 1 - faithfulness01/consistency01
            faith_risk = raw.get("halluc_risk01", None)
        if faith_risk is None:
            # prefer explicit faithfulness01 if present
            faith01 = raw.get("faithfulness01", None)
            if faith01 is None and context:
                # evidence-informed: if provided by upstream retrieval checker
                ev = context.get("evidence") or {}
                faith01 = ev.get("faithfulness01") or ev.get("consistency01")
            faith_risk = 1.0 - float(faith01) if faith01 is not None else 0.5
        faith_risk = self.c_faith(_clamp01(float(faith_risk)))

        # ---- OOD risk ----
        ood = raw.get("ood_hat01", None)
        if ood is None:
            ood = raw.get("ood_risk01", None)
        if ood is None:
            in_dom = raw.get("in_domain01", None)
            if in_dom is None and context:
                in_dom = (context.get("evidence") or {}).get("in_domain01")
            ood = 1.0 - float(in_dom) if in_dom is not None else 0.5
        ood = self.c_ood(_clamp01(float(ood)))

        # ---- Δ gap ----
        delta = raw.get("delta_gap01", None)

        # context-provided direct disagreement wins
        if context and delta is None:
            if "disagreement01" in context:
                delta = float(context["disagreement01"])

        # if a baseline is present, compute Δ as mean L1 distance across risk axes
        if context and delta is None:
            baseline = context.get("baseline_metrics") or context.get("reference_metrics")
            if isinstance(baseline, dict):
                # Normalize/interpret baseline as the same axes; apply same transforms
                b_conf = self._norm_conf_from_baseline(baseline)
                b_faith = self._norm_faith_risk_from_baseline(baseline)
                b_ood = self._norm_ood_from_baseline(baseline)

                # Disagreement is distance between (risk vector) components
                # We compare risk components: [1-conf, faith_risk, ood]
                a_vec = [1.0 - conf, faith_risk, ood]
                b_vec = [1.0 - b_conf, b_faith, b_ood]
                delta = sum(abs(a - b) for a, b in zip(a_vec, b_vec)) / 3.0

        # still none? estimate from uncertainty + ood as a weak proxy
        if delta is None:
            delta = self._estimate_delta_proxy(confidence01=conf, ood_hat01=ood)

        delta = self.c_delta(_clamp01(float(delta)))

        metrics01 = {
            "confidence01": conf,
            "faithfulness_risk01": faith_risk,
            "ood_hat01": ood,
            "delta_gap01": delta,
        }
        return metrics01

    def to_scm(self, goal: str, reply: str, metrics01: Dict[str, float]) -> Dict[str, Any]:
        """
        Convenience mapping to an SCM-style row (extend as needed).
        """
        return scm_row(
            goal=goal,
            reply=reply,
            confidence01=float(metrics01["confidence01"]),
            faithfulness_risk01=float(metrics01["faithfulness_risk01"]),
            ood_hat01=float(metrics01["ood_hat01"]),
            delta_gap01=float(metrics01["delta_gap01"]),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _norm_conf_from_baseline(self, b: Dict[str, Any]) -> float:
        conf = b.get("confidence01")
        if conf is None:
            conf = _invert01(_safe_get(b, "uncertainty01", 0.5))
        return self.c_conf(_clamp01(float(conf)))

    def _norm_faith_risk_from_baseline(self, b: Dict[str, Any]) -> float:
        fr = b.get("faithfulness_risk01")
        if fr is None:
            fr = b.get("halluc_risk01")
        if fr is None:
            faith01 = b.get("faithfulness01")
            if faith01 is None:
                faith01 = b.get("consistency01")
            fr = 1.0 - float(faith01) if faith01 is not None else 0.5
        return self.c_faith(_clamp01(float(fr)))

    def _norm_ood_from_baseline(self, b: Dict[str, Any]) -> float:
        ood = b.get("ood_hat01")
        if ood is None:
            ood = b.get("ood_risk01")
        if ood is None:
            in_dom = b.get("in_domain01")
            ood = 1.0 - float(in_dom) if in_dom is not None else 0.5
        return self.c_ood(_clamp01(float(ood)))

    @staticmethod
    def _estimate_delta_proxy(*, confidence01: float, ood_hat01: float) -> float:
        """
        Weak but monotone proxy for disagreement when only a single model view exists.
        Intuition: higher (1 - confidence) and higher OOD correlate with greater Δ.
        """
        return _clamp01(0.5 * (1.0 - confidence01) + 0.5 * ood_hat01)


__all__ = ["Aligner", "MonotoneCalibrator"]
