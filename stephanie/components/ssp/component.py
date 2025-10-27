from __future__ import annotations

import asyncio
import time
from typing import Optional

from omegaconf import DictConfig

from stephanie.components.ssp.trainer import Trainer
from stephanie.utils.trace_logger import get_trace_logger


class SSPComponent:
    def __init__(self, cfg: DictConfig | dict, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.trainer = Trainer(self.cfg, self.container)
        self.is_running = False
        self._last_tick = None
        self._last_metrics = None
        self._logger = get_trace_logger()

    def start(self, max_steps: Optional[int] = None, background: bool = False, context: Optional[dict] = None):

        self.is_running = True
        steps = 0
        try:
            while self.is_running and (max_steps is None or steps < max_steps):
                # run a full SSP step synchronously to avoid stray coroutines
                self._last_metrics = self.trainer.train_step(context)
                steps += 1
        finally:
            self.is_running = False
        return {"ok": True, "event": "started", "steps": max_steps}

    def stop(self):
        self.is_running = False
        return {"ok": True, "event": "stopping"}

    def tick(self, run_step: bool = False, context: Optional[dict] = None):
        """
        If run_step=True, execute one training step (sync) on each tick.
        Otherwise this is a heartbeat/report-only tick.
        """
        if run_step:
            try:
                self._last_metrics = self.trainer.train_step(context)
            except Exception as e:
                self._last_metrics = {"error": str(e)}

    def status(self):
        return {
            "status": "running" if self.is_running else "idle",
            "episode_count": 0,  # wire if you track it
            "difficulty": float(self.trainer.proposer.difficulty),
            "tick_interval": float(self.cfg.self_play.jitter.tick_interval),
            "last_metrics": self._last_metrics,
            "last_tick": self._last_tick,
        }

    @staticmethod
    def _now_ms():
        import time
        return int(time.time() * 1000)

    async def _ticker_loop(self):
        self.running = True
        while self.running:
            t0 = time.time()
            result = await self.trainer.train_step({})   # <-- await

            self.state.episode_count += 1
            self.state.difficulty = result.get("metrics", {}).get(
                "difficulty", self.trainer.proposer.difficulty
            )
            self.state.last_metrics = result

            delay = max(0.0, float(self.sp.tick_interval) - (time.time() - t0))
            await asyncio.sleep(delay)
