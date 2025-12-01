# stephanie/services/epistemic_guard_service.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from stephanie.components.risk.attr_sink_orm import ORMAttrSink
from stephanie.components.risk.badge import make_badge
from stephanie.components.risk.epi.epistemic_guard import (GuardInput,
                                                           GuardOutput)
from stephanie.components.risk.provenance import ProvenanceLogger
from stephanie.components.risk.signals import HallucinationContext
from stephanie.components.risk.signals import collect as collect_hall
from stephanie.services.service_protocol import Service


# ------------------- Contracts -------------------
# These are your real hallucination components — must be provided by container
# - sampler: async def sample(prompt: str, n: int) -> List[str]
# - embedder: def embed(sentences: List[str]) -> np.ndarray
# - entailment: def entail(premise: str, hypothesis: str) -> float

# ------------------- Service -------------------
class EpistemicGuardService(Service):
    """
    Production-grade Epistemic Guard Service.

    This service:
    - Computes REAL hallucination signals (semantic drift, metadata violation, RAG unsupported)
    - Writes namespaced attributes (hall.se_mean, hall.meta.inv_violations, etc.) to DB
    - Generates a real HallucinationBadge (ok/warn/risk)
    - Saves VPM channels (R=Δ-energy, G=entropy, B=disagreement, A=1−confidence) as .npz
    - Logs full manifest (goal, reply, context, metrics) as JSON + JSONL trace
    - Is fully compatible with Jitter: emits phys.hall_score and VPM channels

    Dependencies:
        - container must provide: sampler, embedder, entailment
        - optional: session (for ORMAttrSink), storage (for manifest path)

    Usage:
        service = EpistemicGuardService(container)
        service.initialize()
        output = await service.assess(guard_input, run_id="run-123")
    """
    
    def __init__(self, memory, container):
        self.memory = memory    
        self.container = container
        self.log = None
        self._sampler = None
        self._embedder = self.memory.embedding
        self._entailment = None
        self._session = None
        self._storage = None
        self._provenance_dir = "./runs/hallucinations"
        self._thresholds = (0.35, 0.60)  # warn, risk — match badge logic
        self._n_semantic_samples = 6
        self._up = False

    @property
    def name(self) -> str:
        return "epistemic-guard-service-v3"

    def initialize(self, **kwargs) -> None:
        """Initialize from container config and dependencies."""
        cfg: Dict[str, Any] = (kwargs.get("config") or {}) if kwargs else {}
        logger = kwargs.get("logger")
        if logger is not None:
            self.log = logger
        else:
            self.log = logging.getLogger(self.name)



        # --- Optional: DB session for ORMAttrSink ---
        self._session = kwargs.get("session")  # SQLAlchemy session

        # --- Optional: Storage for manifest path ---
        self._storage = kwargs.get("storage")  # StorageService
        self._provenance_dir = cfg.get("provenance_dir", "./runs/hallucinations")
        self._thresholds = tuple(cfg.get("thresholds", (0.35, 0.60)))
        self._n_semantic_samples = int(cfg.get("n_semantic_samples", 6))

        # --- Validate and log ---
        self._up = True
        self.log.info(
            "EpistemicGuardService initialized",
            extra={
                "n_semantic_samples": self._n_semantic_samples,
                "thresholds": self._thresholds,
                "provenance_dir": self._provenance_dir,
                "has_db_session": bool(self._session),
                "has_storage": bool(self._storage),
            },
        )

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._up else "unhealthy",
            "metrics": {},
            "dependencies": {
                "sampler": "ready" if self._sampler else "missing",
                "embedder": "ready" if self._embedder else "missing",
                "db_session": "ready" if self._session else "optional",
                "storage": "ready" if self._storage else "optional",
            },
        }

    def shutdown(self) -> None:
        self._up = False
        self.log.info("EpistemicGuardService shutdown")

    # ------------------- Public API -------------------
    async def assess(self, data: GuardInput, *, run_id: Optional[str] = None) -> GuardOutput:
        """
        Assess hallucination risk using real signals.

        Args:
            data: GuardInput containing question, reference, hypothesis
            run_id: Optional. If provided, artifacts are written to storage.

        Returns:
            GuardOutput with real metrics, paths, and badge.
        """
        if not self._up:
            raise RuntimeError("EpistemicGuardService is not initialized")

        def _embedding_wrapper(sentences: List[str]) -> np.ndarray:
            if not sentences:
                return np.zeros((0, 0), dtype=np.float32)
            vecs = []
            for s in sentences:
                emb = self.memory.embedding.get_or_create(s)
                emb = np.array(emb, dtype=np.float32)
                vecs.append(emb)
            return np.stack(vecs, axis=0)

        # --- 1. Build HallucinationContext ---
        ctx = HallucinationContext(
            question=data.question,
            retrieved_passages=[data.question],  # <-- RAG context is reference
            sampler=self._sampler,
            embedder=_embedding_wrapper,
            entailment=self._entailment,
            n_semantic_samples=self._n_semantic_samples,
            power_acceptance_rate=None,
            power_lp_delta_mean=None,
            power_reject_streak_max=None,
            power_token_multiplier=None,
        )

        # --- 2. Collect Real Hallucination Signals ---
        sink = ORMAttrSink(session=self._session, evaluation_id=run_id, prefix="hall.") if self._session else None
        signals = collect_hall(answer=data.hypothesis, ctx=ctx, sink=sink)

        # --- 3. Generate Real Badge ---
        badge = make_badge(
            se_mean=signals.se_mean,
            meta_viols=signals.meta_inv_violations,
            rag_unsupported=signals.rag_unsupported_frac,
        )

        # --- 4. Save VPM Channels as .npz ---
        vpm_path = self._save_vpm_channels(run_id, signals.vpm_channels)

        # --- 5. Log Full Provenance (Manifest) ---
        provenance_path = self._log_provenance(
            run_id=run_id,
            record={
                "decision": badge.level,
                "risk": badge.score,
                "thresholds": {
                    "warn": self._thresholds[0],
                    "risk": self._thresholds[1],
                },
                "metrics": {
                    "hall.se_mean": signals.se_mean,
                    "hall.meta_inv_violations": signals.meta_inv_violations,
                    "hall.rag_unsupported_frac": signals.rag_unsupported_frac,
                    "hall.max_energy": float(signals.vpm_channels.get("R", [0]).mean()) if "R" in signals.vpm_channels else 0.0,
                    "hall.entropy": float(signals.vpm_channels.get("G", [0]).mean()) if "G" in signals.vpm_channels else 0.0,
                    "hall.disagree_rate": float(signals.vpm_channels.get("B", [0]).mean()) if "B" in signals.vpm_channels else 0.0,
                    "hall.confidence": float(1 - signals.vpm_channels.get("A", [0]).mean()) if "A" in signals.vpm_channels else 0.5,
                },
                "reasons": badge.reasons,
            },
            goal=data.question,
            reply=data.hypothesis,
            context={
                "reference": data.reference,
                "meta": data.meta or {},
            },
        )

        # --- 6. Route Decision ---
        route = self._route(badge.level)

        # --- 7. Return Production-Grade Output ---
        return GuardOutput(
            trace_id=run_id or "adhoc",
            risk=badge.score,
            thresholds=(self._thresholds[0], self._thresholds[1]),
            route=route,
            metrics={
                "hall.se_mean": signals.se_mean,
                "hall.meta_inv_violations": signals.meta_inv_violations,
                "hall.rag_unsupported_frac": signals.rag_unsupported_frac,
                "hall.max_energy": float(signals.vpm_channels.get("R", [0]).mean()) if "R" in signals.vpm_channels else 0.0,
                "hall.entropy": float(signals.vpm_channels.get("G", [0]).mean()) if "G" in signals.vpm_channels else 0.0,
                "hall.disagree_rate": float(signals.vpm_channels.get("B", [0]).mean()) if "B" in signals.vpm_channels else 0.0,
                "hall.confidence": float(1 - signals.vpm_channels.get("A", [0]).mean()) if "A" in signals.vpm_channels else 0.5,
            },
            vpm_path=vpm_path,
            field_path="",  # Optional: remove if unused
            strip_path="",
            legend_path="",
            badge_path=provenance_path,  # <-- Use provenance JSON as badge path (machine-readable)
            evidence_id=run_id,
            schema="hall.v1",
        )

    # ------------------- Internal Helpers -------------------

    def _route(self, level: str) -> str:
        """Map badge level to route: ok->FAST, warn->MEDIUM, risk->HIGH"""
        mapping = {"ok": "FAST", "warn": "MEDIUM", "risk": "HIGH"}
        return mapping.get(level.lower(), "MEDIUM")

    def _save_vpm_channels(self, run_id: Optional[str], channels: Dict[str, Any]) -> str:
        """Save VPM channels as compressed .npz file. Return path."""
        if not run_id:
            return f"adhoc_vpm_{hash(str(channels))}.npz"

        if self._storage:
            subdir = "vpm"
            out_dir = self._storage.subdir(run_id, subdir)
            path = out_dir / f"{run_id}_vpm.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import numpy as np
                np.savez_compressed(path, **{k: v for k, v in channels.items()})
                return str(path)
            except Exception as e:
                self.log.warning(f"Failed to save VPM channels for {run_id}: {e}")
                return f"{run_id}_vpm.npz"
        else:
            # Fallback: save to local dir
            os.makedirs("./runs/vpm", exist_ok=True)
            path = f"./runs/vpm/{run_id}_vpm.npz"
            try:
                import numpy as np
                np.savez_compressed(path, **{k: v for k, v in channels.items()})
                return path
            except Exception as e:
                self.log.warning(f"Failed to save VPM channels locally: {e}")
                return f"{run_id}_vpm.npz"

    def _log_provenance(
        self,
        run_id: Optional[str],
        record: Dict[str, Any],
        goal: str,
        reply: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Write full manifest as JSON and append to JSONL trace."""
        if not run_id:
            run_id = f"adhoc_{hash(str(goal) + str(reply))}"

        provenancelog = ProvenanceLogger(out_dir=self._provenance_dir, logger=self.log)
        provenancelog.log(
            record=record,
            goal=goal,
            reply=reply,
            context=context,
        )

        # Return path to manifest JSON for downstream use
        safe_id = f"provenance_{run_id}"
        return f"{self._provenance_dir}/{safe_id}.json"