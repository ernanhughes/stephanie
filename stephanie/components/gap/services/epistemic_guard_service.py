# stephanie/components/gap/services/epistemic_guard_service.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from stephanie.components.gap.processors.epistemic_guard import (
    EGVisual, EpistemicGuard, GuardInput, GuardOutput)
from stephanie.services.service_protocol import Service


class EpistemicGuardService(Service):
    def __init__(self):
        self._logger = logging.getLogger(self.name)
        self._core: EpistemicGuard | None = None
        self._up = False

    @property
    def name(self) -> str:
        return "eg-service-v1"

    def initialize(self, **kwargs) -> None:
        cfg: Dict[str, Any] = (kwargs.get("config") or {}) if kwargs else {}
        logger = kwargs.get("logger")
        if logger is not None:
            self._logger = logger

        out_dir = cfg.get("out_dir", "./static/eg")
        thresholds = tuple(cfg.get("thresholds", (0.2, 0.6)))
        seed = int(cfg.get("seed", 42))

        self._core = EpistemicGuard(out_dir=out_dir, thresholds=thresholds, seed=seed)
        self._up = True
        self._logger.info("EpistemicGuardService initialized", extra={
            "out_dir": out_dir, "thresholds": thresholds, "seed": seed
        })

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._up else "unhealthy",
            "metrics": {},
            "dependencies": {"core": "ready" if self._core else "missing"},
        }

    def shutdown(self) -> None:
        self._up = False
        self._core = None
        self._logger.info("EpistemicGuardService shutdown")

    # -------- public API pass-throughs --------
    async def assess(self, data: GuardInput) -> GuardOutput:
        if not self._core:
            raise RuntimeError("EpistemicGuardService not initialized")
        return await self._core.assess(data)

    def set_predictor(self, predictor: Any) -> None:
        """
        Accepts either:
          - an object with *async* predict_risk(...)
          - or an object with *sync* predict_risk(...)
        and wires it into the EpistemicGuard core.
        """
        if not self._core:
            raise RuntimeError("EpistemicGuardService not initialized")

        pr = getattr(predictor, "predict_risk", None)
        if pr is None:
            raise TypeError("predictor must have a `predict_risk` method")

        # If it's already async, just pass it through
        if asyncio.iscoroutinefunction(pr):
            self._core.set_predictor(predictor)
            return

        # Otherwise wrap a sync predictor into an async adapter
        class _AsyncAdapter:
            async def predict_risk(self, question: str, context: str, **kw):
                return pr(question, context, **kw)

        self._core.set_predictor(_AsyncAdapter())


class EGVisualService(Service):
    def __init__(self):
        self._logger = logging.getLogger(self.name)
        self._visual: EGVisual | None = None
        self._up = False

    @property
    def name(self) -> str:
        return "eg-visual-v1"

    def initialize(self, **kwargs) -> None:
        cfg = (kwargs.get("config") or {}) if kwargs else {}
        logger = kwargs.get("logger")
        if logger is not None:
            self._logger = logger

        out_dir = cfg.get("out_dir", "./static/eg/img")
        seed = int(cfg.get("seed", 42))
        self._visual = EGVisual(out_dir=out_dir, seed=seed)
        self._up = True
        self._logger.info("EGVisualService initialized", extra={"out_dir": out_dir, "seed": seed})

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._up else "unhealthy",
            "metrics": {},
            "dependencies": {"visual": "ready" if self._visual else "missing"},
        }

    def shutdown(self) -> None:
        self._up = False
        self._visual = None
        self._logger.info("EGVisualService shutdown")

    # pass-through
    def render(self, trace_id: str, vpm):
        if not self._visual:
            raise RuntimeError("EGVisualService not initialized")
        return self._visual.render(trace_id, vpm)
