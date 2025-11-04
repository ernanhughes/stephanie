# stephanie/services/daimon/guards.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Optional base import (degrade gracefully if not present)
# -----------------------------------------------------------------------------
try:
    from stephanie.services.daimon.epistemic_guard import EpistemicGuard  # your base class
except Exception:  # pragma: no cover
    class EpistemicGuard:  # minimal shim
        def __init__(self, config: Dict[str, Any] | None = None):
            self.config = config or {}
            # Optional integration points. Inject where available.
            self.action_router = None
            self.memcube = None

# -----------------------------------------------------------------------------
# Risk tier (Enum-like). Keep consistent with rest of Daimon.
# -----------------------------------------------------------------------------
try:
    from stephanie.services.daimon.risk_types import RiskTier  # canonical enum if you have it
except Exception:  # pragma: no cover
    class RiskTier:
        LOW, MEDIUM, HIGH, CRITICAL = range(4)
        name = "RiskTierShim"
        value = LOW

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class StructuralGuardConfig:
    """Thresholds and weights for structural risk assessment."""
    bridge_risk_high: float = 0.60              # vision_bridge_proxy
    symmetry_disagreement: float = 0.40         # |vision_sym - gnn_sym|
    fallback_penalty: float = 0.10              # layout fallback adds baseline risk
    fragile_gap_bucket: int = 0                 # 0 = low, 1 = mid, 2 = high
    fragile_crossings_threshold: int = 10       # crossing edges threshold
    # Weights composing the aggregate risk score (0..1+)
    w_bridge: float = 0.35
    w_disagree: float = 0.30
    w_fallback: float = 0.10
    w_fragile: float = 0.25
    # Tier cutoffs
    tier_critical: float = 0.60
    tier_high: float = 0.40
    tier_medium: float = 0.20


# -----------------------------------------------------------------------------
# Structural Guard
# -----------------------------------------------------------------------------
class StructuralGuard(EpistemicGuard):
    """
    Watches for structural brittleness via vision/GNN signals.
    Triggers: bridge risk spikes, layout fallbacks, vision-GNN disagreement, fragile topology.

    Expected trace_context keys (safe defaults applied if missing):
      {
        "vision_bridge_proxy": float,
        "vision_symmetry": float,
        "gnn_symmetry": float,
        "vision_spectral_gap_bucket": int,   # 0=low,1=mid,2=high
        "layout_fallback": Optional[str],    # e.g., "spring"
        "crossings": int,                    # number of edge crossings (if available)
        "plan_trace": PlanTrace,             # optional, for escalation hooks
        ...
      }
    """

    def __init__(self, config: Dict[str, Any] | StructuralGuardConfig | None = None):
        super().__init__(config if isinstance(config, dict) else None)
        # Normalize config into dataclass + store
        if isinstance(config, StructuralGuardConfig):
            self.cfg = config
        else:
            self.cfg = StructuralGuardConfig(
                **(config or {})
            )

        # Optional integration points (may be injected by container)
        # - action_router: performs mitigation actions
        # - memcube: logs metrics/flags for dashboards
        # They are also present on EpistemicGuard shim; keep attribute names.
        if not hasattr(self, "action_router"):
            self.action_router = None
        if not hasattr(self, "memcube"):
            self.memcube = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assess(self, trace_context: Dict[str, Any]) -> Tuple[RiskTier, List[str]]:
        risk_score, reasons = self._compute_risk(trace_context)
        tier = self._to_tier(risk_score)

        # Telemetry hook (non-fatal if unavailable)
        self._log_assessment(trace_context, risk_score, tier, reasons)

        return tier, reasons

    def escalate(self, trace_id: str, risk_tier: RiskTier, reasons: List[str]) -> None:
        """
        Trigger mitigations when risk is HIGH or CRITICAL.
        Uses your existing action_router and memcube if present.
        """
        try:
            # Persist flag for audit dashboards
            if self.memcube is not None:
                self.memcube.log_flag(
                    trace_id=trace_id,
                    flag=f"STRUCTURE_RISK_{self._tier_name(risk_tier)}",
                    meta={"reasons": reasons},
                )
        except Exception as e:  # pragma: no cover
            _logger.debug("MemCube flag log failed: %s", e)

        if self._is_high(risk_tier) and self.action_router is not None:
            try:
                self.action_router.trigger(
                    actions=[
                        "diversify_plan",         # broaden candidate traces / branches
                        "enable_visual_thought",  # force visual analysis / zoom
                        "escalate_retrieval",     # deepen/document retrieval
                    ],
                    trace_id=trace_id,
                    context={"reasons": reasons, "risk_tier": self._tier_name(risk_tier)},
                )
            except Exception as e:  # pragma: no cover
                _logger.warning("Action router trigger failed: %s", e)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_risk(self, ctx: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Compose a bounded risk score from multiple structural indicators.
        Returns (risk_score, reasons).
        """
        reasons: List[str] = []
        score = 0.0
        cfg = self.cfg

        # 1) Bridge bottleneck risk
        bridge = float(ctx.get("vision_bridge_proxy", 0.0))
        if bridge > cfg.bridge_risk_high:
            score += cfg.w_bridge
            reasons.append(f"high_bridge_risk={bridge:.2f}>={cfg.bridge_risk_high:.2f}")

        # 2) Vision vs GNN disagreement (structural uncertainty)
        v_sym = float(ctx.get("vision_symmetry", 0.5))
        g_sym = float(ctx.get("gnn_symmetry", 0.5))
        disagreement = abs(v_sym - g_sym)
        if disagreement > cfg.symmetry_disagreement:
            score += cfg.w_disagree
            reasons.append(f"sym_disagreement={disagreement:.2f}>={cfg.symmetry_disagreement:.2f}")

        # 3) Layout fallback (degraded perception)
        if ctx.get("layout_fallback"):
            score += cfg.w_fallback
            reasons.append(f"layout_fallback={ctx['layout_fallback']}")

        # 4) Fragile topology: low spectral gap + high crossings
        gap_bucket = int(ctx.get("vision_spectral_gap_bucket", 1))
        crossings = int(ctx.get("crossings", -1))
        if gap_bucket == cfg.fragile_gap_bucket and crossings >= cfg.fragile_crossings_threshold:
            score += cfg.w_fragile
            reasons.append(f"fragile_topology=gap{gap_bucket}_cross{crossings}")

        # Clamp score to [0, 1.0+] just for sanity (not strictly required)
        score = float(max(0.0, min(1.5, score)))
        return score, reasons

    def _to_tier(self, score: float) -> RiskTier:
        cfg = self.cfg
        if score >= cfg.tier_critical:
            return RiskTier.CRITICAL
        if score >= cfg.tier_high:
            return RiskTier.HIGH
        if score >= cfg.tier_medium:
            return RiskTier.MEDIUM
        return RiskTier.LOW

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _is_high(self, tier: RiskTier) -> bool:
        try:
            return tier in (RiskTier.HIGH, RiskTier.CRITICAL)
        except Exception:
            return int(tier) >= int(RiskTier.HIGH)  # shim fallback

    def _tier_name(self, tier: RiskTier) -> str:
        try:
            return getattr(tier, "name", str(int(tier)))
        except Exception:
            return str(int(tier))

    def _log_assessment(
        self,
        ctx: Dict[str, Any],
        score: float,
        tier: RiskTier,
        reasons: List[str],
    ) -> None:
        # Best-effort telemetry to MemCube
        trace_id = str(getattr(ctx.get("plan_trace", None), "id", "unknown"))
        try:
            if self.memcube is not None:
                self.memcube.log_metric("structure_risk_score", score, trace_id=trace_id)
                self.memcube.log_metric("structure_risk_tier", int(tier), trace_id=trace_id)
                if reasons:
                    # Lightweight breadcrumb
                    self.memcube.log_flag(
                        trace_id=trace_id,
                        flag="STRUCTURE_RISK_REASONS",
                        meta={"reasons": reasons},
                    )
        except Exception as e:  # pragma: no cover
            _logger.debug("MemCube telemetry failed: %s", e)


# -----------------------------------------------------------------------------
# Factory (optional): Hydra-style target path convenience
# -----------------------------------------------------------------------------
def build_structural_guard(config: Dict[str, Any] | None = None) -> StructuralGuard:
    """
    Convenience builder if you prefer referencing this in Hydra configs via:
      _target_: stephanie.services.daimon.guards.build_structural_guard
    """
    return StructuralGuard(config=config or {})
