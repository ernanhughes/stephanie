# stephanie/services/daimon/epistemic_guard.py
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from .risk_types import RiskAssessment, RiskTier

log = logging.getLogger(__name__)


# ------------------------------- Integration APIs -------------------------------

class ActionRouter(Protocol):
    """
    Minimal interface for triggering mitigations / escalations.
    Concrete implementation lives in Jitter/Supervisor services.
    """
    def trigger(self, actions: Iterable[str], trace_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        ...


class MemCubeLogger(Protocol):
    """
    Minimal interface for telemetry into MemCube / GAP dashboards.
    """
    def log_metric(self, name: str, value: float, *, trace_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None: ...
    def log_flag(self, *, trace_id: str, flag: str, meta: Optional[Dict[str, Any]] = None) -> None: ...


# --------------------------------- Base Guard ----------------------------------

@dataclass
class EpistemicGuardConfig:
    """
    Common knobs for guards. Extend in concrete guards as needed.
    """
    escalate_on: Tuple[RiskTier, ...] = (RiskTier.HIGH, RiskTier.CRITICAL)
    # Default score→tier thresholds (guards may override)
    threshold_medium: float = 0.20
    threshold_high: float = 0.40
    threshold_critical: float = 0.60


class EpistemicGuard(abc.ABC):
    """
    Abstract base class for Daimon guards. Concrete guards implement `assess()`
    and may override `escalate()` to trigger domain-specific actions.

    Contracts:
      - `assess(context)` returns (RiskTier, reasons[str]) OR RiskAssessment.
      - `escalate(trace_id, tier, reasons)` is best-effort; must not raise.
      - Guards SHOULD log telemetry to MemCube when available.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = EpistemicGuardConfig(**(config or {}))
        self.action_router: Optional[ActionRouter] = None
        self.memcube: Optional[MemCubeLogger] = None

    # ----------------------------- Dependency injection -----------------------------

    def register_action_router(self, router: ActionRouter) -> None:
        self.action_router = router

    def register_memcube(self, memcube: MemCubeLogger) -> None:
        self.memcube = memcube

    # -------------------------------- Core interface --------------------------------

    @abc.abstractmethod
    def assess(self, trace_context: Dict[str, Any]) -> Tuple[RiskTier, List[str]] | RiskAssessment:
        """
        Compute a risk signal from the provided context.

        `trace_context` is an unstructured dict (PlanTrace, scores, metas, etc.).
        Implementations must be robust to missing keys and default sanely.
        """
        raise NotImplementedError

    def escalate(self, trace_id: str, risk_tier: RiskTier, reasons: List[str]) -> None:
        """
        Default escalation is no-op. Concrete guards may override or rely on
        container-injected `action_router` and `memcube`.
        """
        # Best-effort default logging; subclasses may add specific actions.
        try:
            if self.memcube is not None:
                self.memcube.log_flag(
                    trace_id=trace_id,
                    flag=f"RISK_{risk_tier.name}",
                    meta={"reasons": reasons} if reasons else None,
                )
        except Exception as e:  # pragma: no cover
            log.debug("MemCube flag log failed: %s", e)

    # ----------------------------- Orchestration helpers -----------------------------

    def assess_and_maybe_escalate(self, trace_id: str, trace_context: Dict[str, Any]) -> RiskAssessment:
        """
        Convenience helper: assess, log telemetry, and escalate if tier warrants.
        Always returns a `RiskAssessment`.
        """
        result = self.assess(trace_context)
        if isinstance(result, RiskAssessment):
            assessment = result
        else:
            tier, reasons = result
            # Not all guards compute a continuous score; 0.0 is fine.
            assessment = RiskAssessment(tier=tier, score=float(trace_context.get("risk_score", 0.0)),
                                        reasons=tuple(reasons))

        # Telemetry
        self._log_assessment(trace_id, assessment)

        # Escalation policy
        if assessment.tier in self.config.escalate_on:
            try:
                self.escalate(trace_id, assessment.tier, list(assessment.reasons))
            except Exception as e:  # pragma: no cover
                log.warning("Guard escalation failed: %s", e)

        return assessment

    def _log_assessment(self, trace_id: str, assessment: RiskAssessment) -> None:
        try:
            if self.memcube is not None:
                self.memcube.log_metric("risk_score", float(assessment.score), trace_id=trace_id)
                self.memcube.log_metric("risk_tier", float(int(assessment.tier)), trace_id=trace_id,
                                        meta={"tier_label": assessment.tier.label()})
                if assessment.reasons:
                    self.memcube.log_flag(trace_id=trace_id, flag="RISK_REASONS",
                                          meta={"reasons": list(assessment.reasons)})
        except Exception as e:  # pragma: no cover
            log.debug("MemCube metric log failed: %s", e)


# -------------------------------- Guard Registry --------------------------------

class GuardRegistry:
    """
    Lightweight registry to manage guard instances, useful for Hydra-free wiring
    or dynamic enable/disable in Supervisor/Jitter.
    """
    def __init__(self):
        self._guards: Dict[str, EpistemicGuard] = {}

    def register(self, name: str, guard: EpistemicGuard) -> None:
        self._guards[name] = guard

    def get(self, name: str) -> Optional[EpistemicGuard]:
        return self._guards.get(name)

    def assess_all(self, trace_id: str, ctx: Dict[str, Any]) -> Dict[str, RiskAssessment]:
        out: Dict[str, RiskAssessment] = {}
        for name, guard in self._guards.items():
            out[name] = guard.assess_and_maybe_escalate(trace_id, ctx)
        return out
