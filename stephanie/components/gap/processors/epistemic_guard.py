# stephanie/components/gap/processors/epistemic_guard.py
"""
EpistemicGuard — full, self-contained version
---------------------------------------------
Unifies: Risk prediction, HalVis VPM tensors, HRM↔Tiny disagreement overlays,
badge rendering, evidence persistence, and routing advice.


Dependencies (optional-but-detected):
  - numpy, pillow, matplotlib, umap-learn (for stable projections), jsonschema (optional)
Everything else has safe fallbacks.

Author: you
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# --------------------------- Safe imports / light deps --------------------------- #
import numpy as np

from stephanie.components.risk.attr_sink_orm import \
    ORMAttrSink  # For provenance
from stephanie.components.risk.badge import make_badge
from stephanie.components.risk.epi.epistemic_guard import \
    RiskPredictor  # For risk prediction
from stephanie.components.risk.provenance import \
    ProvenanceLogger  # For manifest
# --- ADD THESE IMPORTS AT THE TOP ---
from stephanie.components.risk.signals import HallucinationContext
from stephanie.components.risk.signals import collect as collect_hall

try:
    from PIL import Image
except Exception:
    Image = None  # badge fallbacks will raise if used

try:
    import umap  # stable 2D projection cache
except Exception:
    umap = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

SAFE_ID = re.compile(r"^[a-zA-Z0-9._:-]{3,128}$")


# --------------------------- Subjects (NATS / streams) -------------------------- #
class Subjects:
    VPM_OUT = "vpm.hallucination.{trace_id}"
    ALERTS_OUT = "alerts.hallucination"
    RISK_OUT = "risk.score.{trace_id}"
    BADGE_OUT = "badge.update.{trace_id}"
    INGRESS_QA = "qa.scored"
    INGRESS_TRACE = "trace.completed"
    SCHEMA_VER = "eg.v1"


# --------------------------- Contracts ----------------------------------------- #
@dataclass
class GuardInput:
    trace_id: str
    question: str
    context: str
    reference: str
    hypothesis: str
    # Optional model “views” for disagreement
    hrm_view: Optional[Dict[str, Any]] = None
    tiny_view: Optional[Dict[str, Any]] = None
    # Optional UI/context metrics
    trust: Optional[float] = None
    recency: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None
    domain: Optional[str] = None

    def context_payload(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "reference": self.reference,
            **(self.meta or {}),
        }


@dataclass
class GuardOutput:
    trace_id: str
    risk: float
    thresholds: Tuple[float, float]
    route: str  # FAST | MEDIUM | HIGH
    metrics: Dict[str, Any]
    vpm_path: str
    field_path: str
    strip_path: str
    legend_path: str
    badge_path: str
    evidence_id: Optional[str] = None
    schema: str = Subjects.SCHEMA_VER


# --------------------------- Helpers: safe I/O --------------------------------- #
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _safe_trace_id(s: str) -> str:
    if not SAFE_ID.match(s):
        raise ValueError(f"unsafe trace_id: {s}")
    return s






# --------------------------- Disagreement (HRM↔Tiny) --------------------------- #
def compute_disagreement(
    hrm_view: Optional[Dict[str, Any]],
    tiny_view: Optional[Dict[str, Any]],
    n_tokens: int,
) -> Dict[str, Any]:
    """
    Return a per-token B-channel and summary disagreement metrics.
    Plug in your true alignment / logit margin deltas here.
    Fallback: simple scalar expanded over tokens if logit traces absent.
    """
    if not hrm_view or not tiny_view:
        return {"B": np.zeros(n_tokens), "max_B": 0.0, "disagree_rate": 0.0}

    # Example: use provided 'delta_conf' or 'agree_margin' if present
    max_B = float(
        abs(hrm_view.get("confidence", 0.7) - tiny_view.get("confidence", 0.5))
    )
    disagree_rate = float(tiny_view.get("disagree_rate_spans", 0.0))
    B = np.full(n_tokens, np.clip(max_B, 0, 1), dtype=np.float32)
    return {"B": B, "max_B": max_B, "disagree_rate": disagree_rate}








# --------------------------- Core: EpistemicGuard ------------------------------ #
class EpistemicGuard:
    """
    One-call facade:
      out = await EpistemicGuard(...).assess(GuardInput(...))

    Returns risk, thresholds, route, VPM paths, image paths, and evidence id.
    """

    def __init__(
        self,
        root_out_dir: str = "./runs/risk_visuals",
        thresholds=(0.2, 0.6),
        seed: int = 42,
        *,
        sampler=None,          # <-- REQUIRED: callable(prompt, n) -> List[str]
        embedder=None,         # <-- REQUIRED: callable(List[str]) -> np.ndarray
        entailment=None,       # <-- REQUIRED: callable(a, b) -> float [0,1]
    ):
        self.root = root_out_dir
        self.img_dir = os.path.join(root_out_dir, "img")
        self.vpm_dir = os.path.join(root_out_dir, "vpm")
        _ensure_dir(self.img_dir)
        _ensure_dir(self.vpm_dir)

        self.sampler = sampler          # <-- Store for HallucinationContext
        self.embedder = embedder        # <-- Store for HallucinationContext
        self.entailment = entailment    # <-- Store for HallucinationContext

        # Keep old visuals if needed for UI (optional)
        # self.visual = EGVisual(out_dir=self.img_dir, seed=seed)
        # self.badges = BadgeGenerator(out_dir=self.img_dir)
        # self.store = EvidenceStore(base_dir=self.vpm_dir)

        self.risk = RiskPredictor(thresholds=thresholds)  # <-- This is now DEPRECATED, but keep for backward compat if needed

    def set_predictor(self, predictor) -> None:
        """
        Inject a predictor with an async `predict_risk(question, context, **kw)`
        that returns (risk: float, (low, high)).
        """
        self.risk = predictor

    # ---------- Routing policy ---------- #
    def _route(
        self,
        risk: float,
        low: float,
        high: float,
        max_B: float,
        beta: float = 0.55,
    ) -> str:
        if (risk < low) and (max_B < beta):
            return "FAST"
        if (risk < high) and (max_B < beta):
            return "MEDIUM"
        return "HIGH"

    def _route(self, level: str) -> str:
        """Map badge level to route: ok->FAST, warn->MEDIUM, risk->HIGH"""
        mapping = {"ok": "FAST", "warn": "MEDIUM", "risk": "HIGH"}
        return mapping.get(level.lower(), "MEDIUM")

    def _save_vpm_tensor(self, trace_id: str, vpm_channels: Dict[str, np.ndarray]) -> str:
        """Save VPM channels as a compressed .npz file for Jitter to load."""
        path = os.path.join(self.vpm_dir, f"{trace_id}_vpm.npz")
        # Save all channels as a dict
        np.savez_compressed(path, **{k: v for k, v in vpm_channels.items()})
        return path

    # ---------- Main entry ---------- #
    def assess(self, data: GuardInput) -> GuardOutput:
        trace_id = _safe_trace_id(data.trace_id)
        
        # --- STEP 1: CREATE THE REAL HALLUCINATION CONTEXT ---
        ctx = HallucinationContext(
            question=data.question,
            retrieved_passages=[data.reference],  # <-- This is your RAG context
            sampler=self.sampler,                # <-- YOU MUST PROVIDE THIS
            embedder=self.embedder,              # <-- YOU MUST PROVIDE THIS  
            entailment=self.entailment,          # <-- YOU MUST PROVIDE THIS
            n_semantic_samples=6,
            power_acceptance_rate=None,          # Optional telemetry
            power_lp_delta_mean=None,            # Optional telemetry
            power_reject_streak_max=None,        # Optional telemetry
            power_token_multiplier=None,         # Optional telemetry
        )

        # --- STEP 2: COLLECT REAL HALLUCINATION SIGNALS ---
        # We'll use ORMAttrSink for DB logging, ProvenanceLogger for manifest
        sink = ORMAttrSink(session=None, evaluation_id=trace_id, prefix="hall.")  # We'll fix session later
        signals = collect_hall(answer=data.hypothesis, ctx=ctx, sink=sink)

        # --- STEP 3: GENERATE THE REAL BADGE ---
        badge = make_badge(
            se_mean=signals.se_mean,
            meta_viols=signals.meta_inv_violations,
            rag_unsupported=signals.rag_unsupported_frac,
        )

        # --- STEP 4: LOG PROVENANCE (MANIFEST) ---
        # Create a manifest record for Jitter + audit
        provenance_logger = ProvenanceLogger(out_dir="./runs/hallucinations", logger=self.logger)
        provenance_logger.log(
            record={
                "run_id": trace_id,
                "decision": badge.level,  # "ok", "warn", "risk"
                "risk": badge.score,      # 0.0 to 1.0
                "thresholds": {
                    "warn": 0.35,
                    "risk": 0.60
                },
                "metrics": {
                    "hall.se_mean": signals.se_mean,
                    "hall.meta_inv_violations": signals.meta_inv_violations,
                    "hall.rag_unsupported_frac": signals.rag_unsupported_frac,
                    "hall.vpm_channels": {k: v.tolist() for k, v in signals.vpm_channels.items()},  # For debugging
                },
                "reasons": badge.reasons,
            },
            goal=data.question,
            reply=data.hypothesis,
            context={
                "reference": data.reference,
                "meta": data.meta,
            }
        )

        # --- STEP 5: SAVE VPM TENSOR (for Jitter) ---
        # We need to save the VPM tensor as .npz for Jitter to load later
        vpm_path = self._save_vpm_tensor(trace_id, signals.vpm_channels)

        # --- STEP 6: RENDER VPM VISUALS (optional for UI) ---
        # If you want the field/strip/legend images for your dashboard, use the old EGVisual
        # But we don't need it for Jitter. You can keep it or remove.
        # field_path, strip_path, legend_path = self.visual.render(trace_id, vpm_tensor)  # <-- Optional

        # --- STEP 7: RETURN THE REAL OUTPUT ---
        return GuardOutput(
            trace_id=trace_id,
            risk=badge.score,  # <-- Now this is REAL hallucination risk!
            thresholds=(0.35, 0.60),  # <-- Match your badge thresholds
            route=self._route(badge.level),  # <-- Map "ok"->"FAST", "warn"->"MEDIUM", "risk"->"HIGH"
            metrics={
                "hall.se_mean": signals.se_mean,
                "hall.meta_inv_violations": signals.meta_inv_violations,
                "hall.rag_unsupported_frac": signals.rag_unsupported_frac,
                "hall.max_energy": signals.vpm_channels.get("R", 0).mean() if "R" in signals.vpm_channels else 0.0,
                "hall.entropy": signals.vpm_channels.get("G", 0).mean() if "G" in signals.vpm_channels else 0.0,
                "hall.disagree_rate": signals.vpm_channels.get("B", 0).mean() if "B" in signals.vpm_channels else 0.0,
            },
            vpm_path=vpm_path,
            field_path="",  # Optional, remove if not used
            strip_path="",
            legend_path="",
            badge_path="",  # Optional, remove if not used
            evidence_id=trace_id,  # <-- Links to manifest
        )
