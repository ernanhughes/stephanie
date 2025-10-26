# stephanie/components/ssp/component.py
from __future__ import annotations
import threading, time
from typing import Optional, Dict, Any
from omegaconf import OmegaConf

from stephanie.components.ssp.core import SearchSelfPlayAgent      # or your prior module

class SSPComponent:
    """
    Thin orchestration wrapper around your Self-Play system.
    - starts/stops a background training loop
    - exposes jitter_tick for Jitter substrate
    - safe status reporting for SIS
    """
    def __init__(self, cfg: OmegaConf, logger=None, event_publisher=None):
        self.cfg = cfg
        self.logger = logger
        self.event_publisher = event_publisher
        self._agent = SearchSelfPlayAgent(cfg)  # your existing SSP “component” class
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        self._episodes = 0
        self._last_tick: Optional[Dict[str, Any]] = None
        self._last_metrics: Optional[Dict[str, Any]] = None

    # ---- lifecycle ---------------------------------------------------------
    def start(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        if self._running:
            return self.status()
        self._stop.clear()
        self._running = True

        def _loop():
            try:
                step = 0
                while not self._stop.is_set() and (max_steps is None or step < max_steps):
                    metrics = self._agent.trainer.train_step()
                    self._last_metrics = metrics
                    step += 1
                    self._episodes += 1
                    if self.event_publisher:
                        try:
                            self.event_publisher.publish("ssp.metrics", metrics)
                        except Exception:
                            pass
                    # small cooperative sleep to keep UI responsive
                    time.sleep(0.01)
            finally:
                self._running = False

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> Dict[str, Any]:
        if not self._running:
            return self.status()
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._running = False
        return self.status()

    # ---- substrate / jitter bridge ----------------------------------------
    def tick(self) -> Dict[str, Any]:
        """
        Single ‘jitter’ feed tick. Safe to call from UI.
        """
        out = self._agent.jitter_tick()
        self._last_tick = out
        if self.event_publisher:
            try:
                self.event_publisher.publish("ssp.jitter", out)
            except Exception:
                pass
        return out

    # ---- info --------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        st = self._agent.get_status() if hasattr(self._agent, "get_status") else {}
        return {
            "status": "running" if self._running else "idle",
            "episode_count": self._episodes,
            "difficulty": st.get("current_difficulty", 0.3),
            "tick_interval": st.get("jitter_interval", self.cfg.get("self_play", {}).get("jitter", {}).get("tick_interval", 2.0)),
            "last_metrics": self._last_metrics,
            "last_tick": self._last_tick,
        }
