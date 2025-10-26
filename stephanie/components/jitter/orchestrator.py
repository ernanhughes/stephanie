# stephanie/components/jitter/orchestrator.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from stephanie.components.jitter.lifecycle import JASLifecycleAgent

from stephanie.services.zeromodel_service import ZeroModelService


log = logging.getLogger("stephanie.jas.orchestrator")


class JASOrchestrator:
    """
    Thin, GAP-style orchestrator for Jitter:
      - registers required services (best-effort)
      - loads/merges JAS config
      - constructs and runs the lifecycle engine
      - exposes start/stop and single entrypoint `execute(context)`

    Notes:
      • Keeps JASLifecycleAgent as the “engine” to avoid reimplementation.
      • All long-running logic stays in the lifecycle agent.
      • Orchestrator is the composition & control layer.
    """

    def __init__(self, cfg: Dict[str, Any], container, logger, memory=None):
        self.cfg = dict(cfg or {})
        self.container = container
        self.logger = logger
        self.memory = memory

        self._jas = None            # JASLifecycleAgent
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #
    async def execute(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        One-call “run the system” entrypoint (like GAP).
        Blocks until duration expires or stop() is called.
        """
        context = context or {}
        await self._register_services()
        jas_cfg = await self._load_and_merge_jas_cfg()

        # Build engine
        self._jas = JASLifecycleAgent(
            cfg=jas_cfg,
            memory=self.memory,
            container=self.container,
            logger=self.logger,
        )

        # Initialize
        ok = await self._jas.initialize()
        if not ok:
            return {"status": "error", "message": "JASLifecycleAgent initialization failed"}

        # Run until duration or external stop
        self._running = True
        self._task = asyncio.create_task(self._jas.run(context))

        duration = self.cfg.get("duration")  # seconds (optional)
        try:
            if duration is not None:
                await asyncio.wait_for(self._task, timeout=float(duration))
            else:
                await self._task  # run until internal stop/apoptosis
        except asyncio.TimeoutError:
            # Graceful shutdown by request
            await self.stop()
        except asyncio.CancelledError:
            # External cancel → also shut down
            await self.stop()

        # Collect final status from lifecycle
        try:
            if self._task and self._task.done() and not self._task.cancelled():
                return self._task.result() or {"status": "unknown"}
        except Exception as e:
            log.warning(f"JAS lifecycle returned error: {e}")

        return {"status": "stopped"}

    async def stop(self):
        """Gracefully stop the orchestrated lifecycle."""
        if not self._running:
            return
        self._running = False

        try:
            if self._jas:
                await self._jas.stop()
        finally:
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _register_services(self):
        """
        Best-effort service registration (idempotent). Keeps the orchestrator
        self-contained: if the container already has services, we leave them.
        """
        # ZeroModelService (for all VPM rendering/fields)
        try:
            self.container.get("zeromodel")
        except Exception:
            try:
                self.container.register(
                    name="zeromodel",
                    factory=lambda: ZeroModelService(
                        cfg=self.cfg.get("zero_model_service", {}) or {},
                        memory=self.memory,
                        logger=self.logger,
                    ),
                    dependencies=[],
                    init_args={},
                )
                zms = self.container.get("zeromodel")
                zms.initialize()
                log.debug("Registered & initialized ZeroModelService (v2)")
            except Exception as e:
                log.warning(f"ZeroModelService registration failed (continuing): {e}")

        # If you want SCM or other scoring-adjacent services here, mirror GAP:
        # try:
        #     self.container.get("scm_service")
        # except Exception:
        #     from stephanie.components.gap.services.scm_service import SCMService
        #     try:
        #         self.container.register(
        #             name="scm_service",
        #             factory=lambda: SCMService(),
        #             dependencies=[],
        #             init_args={"config": self.cfg.get("scm", {}), "logger": self.logger},
        #         )
        #     except Exception as e:
        #         log.warning(f"SCMService registration failed (continuing): {e}")

    async def _load_and_merge_jas_cfg(self) -> Dict[str, Any]:
        """ Yes
        Load JAS config file (optional) and merge runtime toggles from orchestrator cfg.
        Keeps shape aligned with current JASLifecycleAgent expectations.
        """
        jas_cfg = {}
        # load from file if provided
        cfg_path = self.cfg.get("jas_config_path") or self.cfg.get("config_path")
        if isinstance(cfg_path, str):
            try:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                jas_cfg = dict(data)
            except Exception as e:
                log.warning(f"Could not load JAS config at {cfg_path}: {e}")

        # ensure top-level keys exist
        jas_cfg.setdefault("core", {})
        jas_cfg.setdefault("triune", {})
        jas_cfg.setdefault("homeostasis", {})
        jas_cfg.setdefault("telemetry", {})
        jas_cfg.setdefault("apoptosis", {})
        jas_cfg.setdefault("reproduction", {})

        # toggles from orchestrator cfg
        if "jas_reproduction_enabled" in self.cfg:
            jas_cfg["reproduction"]["enable_reproduction"] = bool(self.cfg["jas_reproduction_enabled"])

        # wire telemetry interval if provided
        if "telemetry_interval" in self.cfg:
            jas_cfg["telemetry"]["interval"] = float(self.cfg["telemetry_interval"])

        # pass-through VPM scoring (if you enabled it at the orchestrator level),
        # so Apoptosis/VPM logic can see it.
        vpm_scoring = self.cfg.get("vpm_scoring")
        if isinstance(vpm_scoring, dict):
            jas_cfg["apoptosis"]["vpm_scoring"] = dict(vpm_scoring)

        # optional: max runtime guard used by lifecycle (if it supports it)
        if "jas_max_runtime" in self.cfg and self.cfg["jas_max_runtime"] is not None:
            jas_cfg["max_runtime"] = int(self.cfg["jas_max_runtime"])

        return jas_cfg
