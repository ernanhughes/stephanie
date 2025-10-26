# stephanie/components/ssp/substrate.py
from __future__ import annotations

import time
import threading
from dataclasses import asdict
from typing import Optional, Dict, Any
from stephanie.components.ssp.trainer import Trainer
from stephanie.components.ssp.bridge import Bridge
from stephanie.components.ssp.util import (
    get_trace_logger,
    PlanTrace_safe,
    VPMEvolverSafe,
)
import atexit
import threading
from omegaconf import DictConfig
from stephanie.components.ssp.config import ensure_cfg


class SspComponent:
    def __init__(self, cfg: DictConfig | dict):
        cfg = ensure_cfg(cfg)
        self.cfg = cfg
        self.trainer = Trainer(cfg)
        self.proposer = self.trainer.proposer
        self.vpm = VPMEvolverSafe(cfg)
        self.bridge = Bridge(self, cfg)
        self.trace_logger = get_trace_logger()
        self.jitter_interval = float(cfg.self_play.jitter.get("tick_interval", 2.0))
        self._last_tick = 0.0
        self.is_running = False
        self.episode_count = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        atexit.register(self.stop)


    def _run(self, max_steps: Optional[int] = None):
        # Run steps in this thread, but bail when stop() is signalled
        if max_steps is None:
            # continuous
            while not self._stop.is_set():
                try:
                    self.trainer.train_step()
                except Exception:
                    # trainer already logs its own errors; keep looping
                    pass
        else:
            for _ in range(max_steps):
                if self._stop.is_set():
                    break
                self.trainer.train_step()
        self.is_running = False

    def start(self, max_steps: Optional[int] = None, background: bool = True):
        if self.is_running:
            return
        self._stop.clear()
        self.is_running = True
        self.trace_logger.log(PlanTrace_safe(
            trace_id="ssp-comp-start", role="system", goal="SSP start", status="started",
            metadata={"max_steps": max_steps, "background": background},
            input="", output="ssp component activated", artifacts={}
        ))
        if background:
            # IMPORTANT: non-daemon + explicit stop() so we can join cleanly
            self._thread = threading.Thread(
                target=self._run, kwargs={"max_steps": max_steps}, daemon=False
            )
            self._thread.start()
        else:
            self._run(max_steps=max_steps)

    def stop(self, join_timeout: float = 2.0):
        # Graceful shutdown for background mode
        self._stop.set()
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=join_timeout)
            except Exception:
                pass
        self.is_running = False

    def status(self) -> Dict[str, Any]:
        # reflect thread liveness
        if self._thread and not self._thread.is_alive():
            self.is_running = False
        return {
            "status": "running" if self.is_running else "idle",
            "episode_count": self.episode_count,
            "difficulty": float(getattr(self.proposer, "difficulty", 0.0)),
            "tick_interval": self.jitter_interval,
        }


    def tick(self) -> Dict[str, Any]:
        if not self.is_running:
            return {"status": "inactive"}
        now = time.time()
        if now - self._last_tick < self.jitter_interval:
            return {"status": "waiting"}
        self._last_tick = now
        self.episode_count += 1
        bundle = self.bridge.get_sensory()
        return {"status": "ok", "sensory": asdict(bundle)}

    def status(self) -> Dict[str, Any]:
        return {
            "status": "running" if self.is_running else "idle",
            "episode_count": self.episode_count,
            "difficulty": getattr(self.proposer, "difficulty", 0.0),
            "tick_interval": self.jitter_interval,
        }
