# stephanie/components/ssp/core.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from stephanie.components.ssp.trainer import Trainer
from stephanie.components.ssp.util import (PlanTrace_safe, VPMEvolverSafe,
                                           get_trace_logger)
from stephanie.utils.json_sanitize import sanitize


class SearchSelfPlayAgent:
    """
    Minimal adaptor so SSPComponent can call:
      - trainer.train_step()
      - jitter_tick()
      - get_status()

    Internally reuses your existing Trainer + VPMEvolverSafe.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.trace_logger = get_trace_logger()

        # Core building blocks already in your repo
        self.trainer = Trainer(cfg)       # has proposer / solver / verifier inside
        self.vpm = VPMEvolverSafe(cfg)    # used by Bridge & ticks

        # Nice to have: expose proposer for difficulty/status
        self.proposer = self.trainer.proposer

        # Jitter cadence
        # (we accept either self_play.jitter.tick_interval or ssp.jitter.tick_interval_sec)
        sp = getattr(cfg, "self_play", None) or getattr(cfg, "ssp", None)
        if sp and hasattr(sp, "jitter"):
            # support both keys
            self.jitter_interval = float(
                sp.jitter.get("tick_interval", sp.jitter.get("tick_interval_sec", 2.0))
            )
        else:
            self.jitter_interval = 2.0

        self._last_tick = 0.0
        self._episode = 0

    # ---------- substrate feed ----------
    def jitter_tick(self) -> Dict[str, Any]:
        now = time.time()
        if (now - self._last_tick) < self.jitter_interval:
            return {"status": "waiting"}

        self._last_tick = now
        self._episode += 1

        # Pull current visual state (safe numpy → JSON handled by sanitize)
        vpm_state = self.vpm.get_current_state()
        scm = {
            "coherence": 0.74,
            "novelty": 0.61,
            "complexity": 0.48,
        }
        # success rate from trainer
        sr = self.trainer.success_history
        success_rate = (sum(sr) / len(sr)) if sr else 0.0
        epistemic = {
            "current_difficulty": getattr(self.proposer, "difficulty", 0.3),
            "recent_success_rate": success_rate,
            "knowledge_growth": min(1.0, max(0.0, success_rate * 0.8 + 0.1)),
            "verification_rate": success_rate,
        }

        payload = {
            "vpm": vpm_state,
            "scm": scm,
            "epistemic": epistemic,
            "meta": {
                "episode": self._episode,
                "tick_interval": self.jitter_interval,
                "timestamp": now,
            },
        }

        # Trace (JSON-safe)
        self.trace_logger.log(PlanTrace_safe(
            trace_id=f"jitter-bridge-{int(now*1000)%1000000}",
            role="jitter",
            goal="substrate feed",
            status="completed",
            metadata={"channels": ["vpm", "scm", "epistemic"]},
            input="",
            output="tick",
            artifacts=sanitize(payload),
        ))
        return sanitize(payload)

    # ---------- status ----------
    def get_status(self) -> Dict[str, Any]:
        sr = self.trainer.success_history
        success_rate = (sum(sr) / len(sr)) if sr else 0.0
        return {
            "current_difficulty": float(getattr(self.proposer, "difficulty", 0.3)),
            "jitter_interval": self.jitter_interval,
            "episode_count": self._episode,
            "recent_success_rate": success_rate,
        }
