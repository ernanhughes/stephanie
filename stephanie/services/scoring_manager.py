# stephanie/services/scoring_manager.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import networkx as nx

from stephanie.data.plan_trace import \
    PlanTrace  # adjust import if your path differs
from stephanie.services.graph_vision_scorer import VisionScorer

# Daimon (risk) is optional; we degrade gracefully if not wired
try:
    from stephanie.services.daimon.guards import (  # your patch earlier
        RiskTier, StructuralGuard)
except Exception:  # pragma: no cover
    StructuralGuard = object  # type: ignore

    class RiskTier:  # minimal shim
        LOW, MEDIUM, HIGH, CRITICAL = range(4)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScoringManagerConfig:
    enable_vision: bool = True
    enable_daimon_guard: bool = True
    # Vision fusion weights (probe-validated defaults)
    vision_symmetry_blend: float = 0.30   # portion added to gnn_symmetry
    base_symmetry_weight: float = 0.70    # retained portion for existing gnn_symmetry
    # Bridge channel key
    bridge_channel_key: str = "bridge_risk"
    # Risk thresholds (also set in configs/daimon.yaml, kept here for convenience)
    bridge_risk_high: float = 0.60
    symmetry_disagreement: float = 0.40
    fallback_penalty: float = 0.10


# ---------------------------------------------------------------------------
# Scoring Manager
# ---------------------------------------------------------------------------

class ScoringManager:
    """
    Aggregates multi-source scores (MRQ/EBT/SICQL/LLM/…),
    fuses vision-based structural signals, and routes risk to Daimon.
    """

    def __init__(
        self,
        cfg: Optional[ScoringManagerConfig] = None,
        vision_scorer: Optional[VisionScorer] = None,
        memcube: Optional[MemCubeService] = None,
        daimon_structural_guard: Optional[StructuralGuard] = None,
    ) -> None:
        self.cfg = cfg or ScoringManagerConfig()
        self.vision_scorer = vision_scorer or VisionScorer(model_path=None, device="cpu")
        self.memcube = memcube
        self.daimon_structural_guard = daimon_structural_guard

    # ----------------------------- Public API -----------------------------

    def score_plan_trace(self, plan_trace: PlanTrace) -> Dict[str, float]:
        """
        Returns the fused score dictionary for a single PlanTrace.
        Side effects: logs to MemCube; triggers Daimon escalation when needed.
        """
        # 1) Base scores from the existing stack
        base_scores = self.compute_base_scores(plan_trace)

        # 2) Vision scores (cached by VisionScorer using plan_trace.id)
        vision_scores: Dict[str, float] = {}
        if self.cfg.enable_vision and hasattr(plan_trace, "graph") and isinstance(plan_trace.graph, nx.Graph):
            try:
                vision_scores = self.vision_scorer.score_graph(
                    plan_trace.graph,
                    cache_key=str(getattr(plan_trace, "id", None) or "trace")
                )
            except Exception as e:
                log.warning("Vision scoring failed: %s", e)

        # 3) Blend vision into base scores (lightweight + safe)
        fused = self.fuse_vision_scores(base_scores, vision_scores)

        # 4) Daimon structural guard (risk) — optional, but recommended
        if self.cfg.enable_daimon_guard and self.daimon_structural_guard is not None:
            try:
                trace_ctx = self._build_structural_context(plan_trace, fused, vision_scores)
                tier, reasons = self.daimon_structural_guard.assess(trace_ctx)  # type: ignore[attr-defined]
                self._log_risk(plan_trace, tier, reasons)

                # Trigger actions if high risk
                if hasattr(self.daimon_structural_guard, "escalate"):
                    self.daimon_structural_guard.escalate(  # type: ignore[attr-defined]
                        trace_id=str(getattr(plan_trace, "id", "unknown")),
                        risk_tier=tier,
                        reasons=reasons or [],
                    )
            except Exception as e:
                log.warning("Structural guard assess/escalate failed: %s", e)

        return fused

    # ---------------------------- Base Scoring ----------------------------

    def compute_base_scores(self, plan_trace: PlanTrace) -> Dict[str, float]:
        """
        Adapter into your existing scoring stack.
        Replace this stub with calls into MRQ/EBT/SICQL/LLM, etc.
        MUST return a flat dict of scores (floats).
        """
        # Example placeholders. Replace with real stack calls.
        scores: Dict[str, float] = {
            "gnn_symmetry": getattr(plan_trace.meta, "gnn_symmetry", 0.5) if hasattr(plan_trace, "meta") else 0.5,
            "sicql_q": getattr(plan_trace.meta, "sicql_q", 0.0) if hasattr(plan_trace, "meta") else 0.0,
            "mrq_score": getattr(plan_trace.meta, "mrq_score", 0.0) if hasattr(plan_trace, "meta") else 0.0,
        }
        return scores

    # ---------------------------- Vision Fusion ---------------------------

    def fuse_vision_scores(self, base_scores: Dict[str, float], vision_scores: Dict[str, Any]) -> Dict[str, float]:
        """
        Lightweight fusion: reinforce symmetry with vision and add a bottleneck channel.
        Safe if vision_scores is empty.
        """
        fused = dict(base_scores)

        if not vision_scores:
            return fused

        v_sym = float(vision_scores.get("vision_symmetry", 0.0))
        gnn_sym = float(base_scores.get("gnn_symmetry", 0.5))

        # Symmetry blend (70/30 per prior probe validation)
        #   gnn_symmetry = 0.7 * old + 0.3 * vision_symmetry
        fused["gnn_symmetry"] = (
            self.cfg.base_symmetry_weight * gnn_sym
            + self.cfg.vision_symmetry_blend * v_sym
        )

        fused[self.cfg.bridge_channel_key] = float(vision_scores.get("vision_bridge_proxy", 0.0))

        # Feed-through spectral gap bucket (integer)
        fused["vision_spectral_gap_bucket"] = int(vision_scores.get("vision_spectral_gap_bucket", 1))

        # Diagnostics (optional deltas)
        fused["vision_symmetry_delta"] = fused["gnn_symmetry"] - gnn_sym

        # MemCube log delta for audit
        if self.memcube is not None:
            try:
                self.memcube.log_metric("vision_fusion_delta", fused["vision_symmetry_delta"])
            except Exception as e:
                log.debug("MemCube log_metric failed: %s", e)

        return fused

    # ---------------------------- Risk Wiring -----------------------------

    def _build_structural_context(
        self,
        plan_trace: PlanTrace,
        fused: Dict[str, float],
        vision_scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Consolidate all structural signals for Daimon.
        """
        ctx: Dict[str, Any] = {
            # Primary fused channels
            "gnn_symmetry": float(fused.get("gnn_symmetry", 0.5)),
            "vision_symmetry": float(vision_scores.get("vision_symmetry", 0.5)),
            "vision_bridge_proxy": float(vision_scores.get("vision_bridge_proxy", 0.0)),
            "vision_spectral_gap_bucket": int(vision_scores.get("vision_spectral_gap_bucket", 1)),
            # Optional metadata
            "layout_fallback": None,
            "crossings": -1,
            "plan_trace": plan_trace,
        }

        # If per-layout is present, propagate any fallback markers
        per_layout = vision_scores.get("per_layout", []) if isinstance(vision_scores, dict) else []
        if per_layout:
            # mark fallback if any layout fell back (e.g., FA2→spring)
            fallbacks = [p.get("fallback") for p in per_layout if p.get("fallback")]
            ctx["layout_fallback"] = fallbacks[0] if fallbacks else None

        # If your PlanTrace meta includes crossings (from layout service), attach it
        if hasattr(plan_trace, "meta") and isinstance(plan_trace.meta, dict):
            ctx["crossings"] = int(plan_trace.meta.get("crossings", -1))
            if "layout_fallback" in plan_trace.meta and plan_trace.meta["layout_fallback"]:
                ctx["layout_fallback"] = plan_trace.meta["layout_fallback"]

        return ctx

    def _log_risk(self, plan_trace: PlanTrace, tier: RiskTier, reasons: Optional[list[str]]) -> None:
        """
        Persist structured risk telemetry for Gap/SIS dashboards.
        """
        if self.memcube is None:
            return

        trace_id = str(getattr(plan_trace, "id", "unknown"))

        try:
            self.memcube.log_metric("structure_risk_tier", int(tier), trace_id=trace_id)
        except Exception as e:
            log.debug("MemCube log_metric(structure_risk_tier) failed: %s", e)

        if reasons:
            try:
                self.memcube.log_flag(
                    trace_id=trace_id,
                    flag=f"STRUCTURE_RISK_{self._tier_name(tier)}",
                    meta={"reasons": reasons},
                )
            except Exception as e:
                log.debug("MemCube log_flag(STRUCTURE_RISK_*) failed: %s", e)

    @staticmethod
    def _tier_name(tier: RiskTier) -> str:
        try:
            # if Enum
            return getattr(tier, "name", str(int(tier)))
        except Exception:
            return str(int(tier))
