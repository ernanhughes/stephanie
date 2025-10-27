from __future__ import annotations
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

    def start(self, max_steps: Optional[int] = None, background: bool = False):
        self.is_running = True
        steps = 0
        try:
            while self.is_running and (max_steps is None or steps < max_steps):
                self._last_metrics = self.trainer.train_step()
                steps += 1
        finally:
            self.is_running = False
        return {"ok": True, "event": "started", "steps": max_steps}

    def stop(self):
        self.is_running = False
        return {"ok": True, "event": "stopping"}

    def tick(self):
        self._last_tick = {
            "status": "ok",
            "time_ms": self._now_ms(),
            "metrics": self._last_metrics,
        }
        return self._last_tick

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
