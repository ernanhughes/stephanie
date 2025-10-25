"""
JAS Lifecycle Agent – Tick-based service that drives the AutopoieticCore
and publishes telemetry to the Arena bus.

Requirements:
- stephanie.bus.nats_client.get_js() → JetStream (async) client
- cfg with:
    tick_interval: float seconds
    subjects: { telemetry, events, death }
    membrane: {...}
    energy:   {...}
    homeostasis: {...}
"""

from __future__ import annotations
import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional

import torch

from .jas_core import AutopoieticCore
from .jas_homeostasis import Homeostasis
from stephanie.services.bus.nats_client import get_js  # async JetStream helper

log = logging.getLogger("stephanie.jas.lifecycle")


class JitterLifecycleAgent:
    def __init__(self, cfg: Dict[str, Any], ebt_model, vpm_manager=None, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or log

        # Subjects
        subjects = cfg.get("subjects", {})
        self.SUB_TELEM = subjects.get("telemetry", "arena.jitter.telemetry")
        self.SUB_EVENTS = subjects.get("events", "arena.jitter.events")
        self.SUB_DEATH = subjects.get("death", "arena.jitter.death")

        # Core + Homeostasis
        self.core = AutopoieticCore(cfg, ebt_model=ebt_model, vpm_manager=vpm_manager, logger=self.logger)
        self.core.attach_homeostasis(Homeostasis(cfg.get("homeostasis", {})))

        # Runtime
        self.tick = 0
        self._js = None
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    # ---------------------------- lifecycle ---------------------------------
    async def start(self) -> None:
        self._js = await get_js()
        self._task = asyncio.create_task(self._loop())
        self.logger.info("🧬 JitterLifecycleAgent started")

    async def stop(self) -> None:
        self._stopped.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("🛑 JitterLifecycleAgent stopped")

    # ----------------------------- loop -------------------------------------
    async def _loop(self) -> None:
        interval = float(self.cfg.get("tick_interval", 1.0))
        try:
            while not self._stopped.is_set():
                await self._tick_once()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        except Exception as e:
            self.logger.exception(f"Lifecycle loop error: {e}")

    async def _tick_once(self) -> None:
        self.tick += 1

        # 1) Gather sensory input (replace with real stimuli when ready)
        sensory = self._get_sensory_embedding()

        # 2) Run autopoietic cycle
        vitals = self.core.cycle(sensory)

        # 3) Prepare telemetry
        payload = {
            "type": "jas_telemetry",
            "tick": self.tick,
            "ts": time.time(),
            "data": {
                "status": vitals["status"],
                "boundary_integrity": vitals["boundary_integrity"],
                "boundary_thickness": vitals["boundary_thickness"],
                "stress": vitals["stress"],
                "delta_e": vitals["delta_e"],
                "homeo_correction": vitals["homeo_correction"],
                "alive": vitals["alive"],
                "energy": vitals["energy"],  # dict(cognitive/metabolic/reserve)
            },
        }

        # 4) Publish telemetry
        await self._js.publish(self.SUB_TELEM, payload)

        # 5) Events: death or special transitions
        if not vitals["alive"]:
            await self._js.publish(self.SUB_DEATH, json.dumps(payload).encode())
            self.logger.warning("💀 Jitter energy depleted. Stopping.")
            await self.stop()

    # --------------------------- sensory stub --------------------------------
    def _get_sensory_embedding(self) -> torch.Tensor:
        """
        Replace with: embeddings from SSP, ATS traces, GAP maps, etc.
        For now: random unit-normalized 1024-dim vector (1x1024 tensor).
        """
        x = torch.randn(1024)
        x = x / (torch.norm(x) + 1e-8)
        return x.unsqueeze(0)
