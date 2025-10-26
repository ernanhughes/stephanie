from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.jitter.orchestrator import JASOrchestrator

log = logging.getLogger("stephanie.agents.jas_pipeline")


@dataclass
class JASPipelineConfig:
    """
    Ultra-thin agent config. Everything else is handled by the orchestrator.
    """
    # lifecycle
    duration: Optional[int] = None
    auto_start: bool = True
    graceful_shutdown_timeout: float = 30.0

    # JAS file/config
    jas_config_path: str = "conf/agent/jitter_v1.yaml"
    jas_reproduction_enabled: bool = True
    jas_max_runtime: Optional[int] = None

    # telemetry (forwarded)
    telemetry_interval: float = 1.0
    health_check_interval: float = 10.0
    max_offspring: int = 5              # Maximum number of offspring to create
    legacy_preservation: bool = True     # Preserve legacy on shutdown

    # VPM scoring passthrough (into JAS apoptosis/triune)
    vpm_scoring: Dict[str, Any] = field(default_factory=dict)

    # ZeroModel service opts (optional)
    zero_model_service: Dict[str, Any] = field(default_factory=dict)


class JASPipelineAgent(BaseAgent):
    """
    Tin husk → defers to JASOrchestrator (GAP-style).
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.config = self._parse_config(cfg)
        self._orch = JASOrchestrator(
            cfg={
                "duration": self.config.duration,
                "jas_config_path": self.config.jas_config_path,
                "jas_reproduction_enabled": self.config.jas_reproduction_enabled,
                "jas_max_runtime": self.config.jas_max_runtime,
                "telemetry_interval": self.config.telemetry_interval,
                "vpm_scoring": self.config.vpm_scoring,
                "zero_model_service": self.config.zero_model_service,
            },
            container=self.container,
            logger=self.logger,
            memory=self.memory,
        )
        self._running = False

    def _parse_config(self, cfg: Dict[str, Any]) -> JASPipelineConfig:
        if isinstance(cfg, dict):
            return JASPipelineConfig(**cfg)
        return JASPipelineConfig()

    async def run(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Single-call entrypoint; all real work is done in the orchestrator.
        """
        self._running = True
        try:
            return await self._orch.execute(context or {})
        finally:
            self._running = False

    async def stop(self):
        await self._orch.stop()
