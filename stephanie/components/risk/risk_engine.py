# stephanie/components/gap/risk/risk_engine.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class _Thresholds:
    faithfulness: float = 0.35    # higher = riskier (faithfulness_risk01)
    uncertainty: float = 0.40     # applied to (1 - confidence01)
    ood: float = 0.30             # higher = riskier (ood_hat01)
    delta: float = 0.30           # higher = riskier (delta_gap01)


def _clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return float(x)


class RiskEngine:
    """
    Threshold-based risk decision with hysteresis + reason codes.

    Input: metrics01 dict:
      - confidence01          (higher -> safer)
      - faithfulness_risk01   (higher -> riskier)
      - ood_hat01             (higher -> riskier)
      - delta_gap01           (higher -> riskier)

    Output: (decision, reasons)
      decision in {"OK", "WATCH", "RISK"}
      reasons:
        - risk_faith : float (faithfulness_risk01)
        - risk_ood   : float (ood_hat01)
        - risk_delta : float (delta_gap01)
        - risk_unc   : float (1 - confidence01)
    """

    def __init__(
        self,
        thresholds: Dict[str, float],
        hysteresis: float = 0.05,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.th = _Thresholds(
            faithfulness=float(thresholds.get("faithfulness", 0.35)),
            uncertainty=float(thresholds.get("uncertainty", 0.40)),
            ood=float(thresholds.get("ood", 0.30)),
            delta=float(thresholds.get("delta", 0.30)),
        )
        self.hysteresis = float(hysteresis)
        self.logger = logger or logging.getLogger(__name__)
        self._last_decision: Optional[str] = None

    # ------------------------------------------------------------------ #
    def thresholds_dict(self) -> Dict[str, float]:
        return dict(
            faithfulness=self.th.faithfulness,
            uncertainty=self.th.uncertainty,
            ood=self.th.ood,
            delta=self.th.delta,
        )

    # ------------------------------------------------------------------ #
    def decide(self, m: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        # compute risk contributors
        risk_unc = _clamp01(1.0 - float(m.get("confidence01", 0.5)))
        risk_faith = _clamp01(float(m.get("faithfulness_risk01", 0.5)))
        risk_ood = _clamp01(float(m.get("ood_hat01", 0.5)))
        risk_delta = _clamp01(float(m.get("delta_gap01", 0.5)))

        flags = [
            risk_faith >= self.th.faithfulness,
            risk_ood   >= self.th.ood,
            risk_delta >= self.th.delta,
            risk_unc   >= self.th.uncertainty,
        ]
        risk_sum = int(sum(1 for f in flags if f))

        # Base decision
        if risk_sum >= 2:
            base = "RISK"
        elif risk_sum == 1:
            base = "WATCH"
        else:
            base = "OK"

        # Hysteresis: avoid flip-flop unless margin clears the band
        if self._last_decision and self._last_decision != base:
            margin = max(
                risk_faith - self.th.faithfulness,
                risk_ood   - self.th.ood,
                risk_delta - self.th.delta,
                risk_unc   - self.th.uncertainty,
            )
            if margin < self.hysteresis:
                base = self._last_decision

        self._last_decision = base

        reasons = dict(
            risk_faith=risk_faith,
            risk_ood=risk_ood,
            risk_delta=risk_delta,
            risk_unc=risk_unc,
        )
        return base, reasons
