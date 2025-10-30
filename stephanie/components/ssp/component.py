# stephanie/components/ssp/component.py
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, Optional
from omegaconf import DictConfig

from .core.curriculum import QMaxCurriculum
from .trainer import Trainer  # if you have a Trainer; otherwise inject externally

log = logging.getLogger("stephanie.ssp.component")

class SSPComponent:
    """
    Orchestrates proposer/solver/verifier via a Trainer (or injected services).
    Exposes:
      - start()/stop()
      - status(): stable dict for dashboards
    """
    def __init__(self, cfg: DictConfig | dict, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.trainer = Trainer(self.cfg, self.container)
        self.curriculum = QMaxCurriculum(
            window=cfg.get("curriculum", {}).get("window", 200),
            target_success=cfg.get("curriculum", {}).get("target_success", 0.65),
        )
        self._status: Dict[str, Any] = {
            "ticks": 0,
            "last_tick_ms": 0.0,
            "last_ok": True,
            "curriculum": self.curriculum.snapshot(),
        }
        self._ticker_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ticker_task = asyncio.create_task(self._ticker_loop())
        log.info("SSPComponent started")

    async def stop(self) -> None:
        self._running = False
        if self._ticker_task:
            self._ticker_task.cancel()
            try:
                await self._ticker_task
            except asyncio.CancelledError:
                pass
        log.info("SSPComponent stopped")

    async def _ticker_loop(self) -> None:
        interval = float(self.cfg.get("tick_interval", 1.0))
        while self._running:
            started = time.perf_counter()
            ok = True
            try:
                if self.trainer and hasattr(self.trainer, "tick"):
                    # Expect: ret = await trainer.tick() -> dict with {return, success}
                    ret = await asyncio.wait_for(self.trainer.tick(), timeout=max(5.0, interval * 5))
                    if isinstance(ret, dict) and "return" in ret and "success" in ret:
                        self.curriculum.update(ret["return"], bool(ret["success"]))
            except Exception as e:
                ok = False
                log.exception("SSP tick failed: %s", e)
            finally:
                elapsed = (time.perf_counter() - started) * 1000.0
                self._status.update({
                    "ticks": self._status["ticks"] + 1,
                    "last_tick_ms": elapsed,
                    "last_ok": ok,
                    "curriculum": self.curriculum.snapshot(),
                })
            await asyncio.sleep(max(0.0, interval))

    def status(self) -> Dict[str, Any]:
        """Stable status dict for dashboards / health checks."""
        return dict(self._status)
