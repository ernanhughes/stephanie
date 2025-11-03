# stephanie/components/jitter/sensors/halluc_sensor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from stephanie.components.risk.badge import make_badge
from stephanie.components.risk.signals import HallucinationContext
from stephanie.components.risk.signals import collect as collect_hall
from stephanie.services.workers.vpm_worker import VPMWorkerInline


@dataclass
class HallucSensorConfig:
    enabled: bool = True
    n_semantic_samples: int = 4

class HallucinationSensor:
    def __init__(
        self,
        cfg: HallucSensorConfig,
        bus,
        vpm_worker: VPMWorkerInline,
        sampler,
        embedder,
        entailment,
    ):
        self.cfg = cfg
        self.bus = bus
        self.vpm: VPMWorkerInline = vpm_worker
        self.sampler = sampler
        self.embedder = embedder
        self.entailment = entailment

    def scan(
        self,
        question: str,
        answer: str,
        retrieved: Optional[List[str]],
        telemetry: Dict[str, float],
        tag: str,
    ) -> Optional[float]:
        if not self.cfg.enabled:
            return None

        ctx = HallucinationContext(
            question=question,
            retrieved_passages=retrieved or [],
            sampler=self.sampler,
            embedder=self.embedder,
            entailment=self.entailment,
            n_semantic_samples=self.cfg.n_semantic_samples,
            power_acceptance_rate=telemetry.get("ps.accept_rate"),
            power_lp_delta_mean=telemetry.get("ps.lp_delta_mean"),
            power_reject_streak_max=telemetry.get("ps.reject_streak_max"),
            power_token_multiplier=telemetry.get("ps.token_mult"),
        )

        signals = collect_hall(answer, ctx, sink=None)  # No DB write
        badge = make_badge(
            signals.se_mean,
            signals.meta_inv_violations,
            signals.rag_unsupported_frac,
        )

        # Stream to Jitter bus
        self.bus.publish("jitter.sensor.hall", {
            "tag": tag,
            "badge": badge.level,
            "score": badge.score,
            "reasons": badge.reasons,
            "channels": {k: v.tolist() for k, v in signals.vpm_channels.items()},
        })

        # Overlay on VPM
        if self.vpm:
            self.vpm.add_channels(tag, signals.vpm_channels, namespace="hall")

        # Return scalar for physiology line
        return badge.score