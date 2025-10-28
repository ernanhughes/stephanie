# stephanie/components/risk/services/epistemic_guard_service.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from stephanie.components.risk.epi.epistemic_guard import EGVisual, EpistemicGuard, GuardInput, GuardOutput
from stephanie.services.service_protocol import Service

# Optional: your storage API (generic, lives under GAP but is reusable)
from stephanie.services.storage import GapStorageService  # noqa: F401


class EpistemicGuardService(Service):
    """
    Run-scoped EpistemicGuard wrapper.

    - If a run_id is provided (recommended), artifacts are written under:
        <storage.base>/<run_id>/visuals/risk/
      via your GapStorageService.
    - Otherwise, falls back to a static out_dir (useful for ad-hoc tests).
    """

    def __init__(self):
        self._logger = logging.getLogger(self.name)
        self._core: EpistemicGuard | None = None   # static fallback
        self._storage: Optional[GapStorageService] = None
        self._visuals_subdir = "visuals/risk"
        self._static_out_dir: Optional[str] = None
        self._thresholds = (0.2, 0.6)
        self._seed = 42
        self._up = False

    @property
    def name(self) -> str:
        return "eg-service-v2"

    def initialize(self, **kwargs) -> None:
        cfg: Dict[str, Any] = (kwargs.get("config") or {}) if kwargs else {}
        logger = kwargs.get("logger")
        if logger is not None:
            self._logger = logger

        # storage is optional but preferred for run-scoped writes
        self._storage = kwargs.get("storage")  # GapStorageService or compatible
        self._visuals_subdir = cfg.get("visuals_subdir", "visuals/risk")
        self._static_out_dir = cfg.get("out_dir")  # only used if no storage/run_id
        self._thresholds = tuple(cfg.get("thresholds", (0.2, 0.6)))
        self._seed = int(cfg.get("seed", 42))

        # Fallback static core (only used when no run_id is provided)
        if self._static_out_dir:
            self._core = EpistemicGuard(
                root_out_dir=self._static_out_dir,
                thresholds=self._thresholds,
                seed=self._seed,
            )

        self._up = True
        self._logger.info(
            "EpistemicGuardService initialized",
            extra={
                "visuals_subdir": self._visuals_subdir,
                "static_out_dir": self._static_out_dir,
                "thresholds": self._thresholds,
                "seed": self._seed,
                "has_storage": bool(self._storage),
            },
        )

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._up else "unhealthy",
            "metrics": {},
            "dependencies": {
                "storage": "ready" if self._storage else "none",
                "static_core": "ready" if self._core else "none",
            },
        }

    def shutdown(self) -> None:
        self._up = False
        self._core = None
        self._logger.info("EpistemicGuardService shutdown")

    # -------- public API --------

    async def assess(self, data: GuardInput, *, run_id: Optional[str] = None) -> GuardOutput:
        """
        Assess risk and render artifacts.

        - If run_id is provided AND storage is configured, render directly into:
            storage.subdir(run_id, visuals_subdir)
        - Else, use the static core/out_dir configured at initialize().
        """
        if self._storage and run_id:
            out_root = self._storage.subdir(run_id, self._visuals_subdir)
            core = EpistemicGuard(
                root_out_dir=str(out_root),
                thresholds=self._thresholds,
                seed=self._seed,
            )
            return await core.assess(data)

        if not self._core:
            raise RuntimeError(
                "EpistemicGuardService not initialized with static out_dir and no run_id/storage provided."
            )
        return await self._core.assess(data)

    def set_predictor(self, predictor: Any) -> None:
        """
        Accepts either:
          - an object with *async* predict_risk(...)
          - or an object with *sync* predict_risk(...)
        Wires predictor into the static core (used when no run_id is provided).
        For run-scoped calls with storage+run_id, pass a predictor in your orchestrator instead.
        """
        if not self._core:
            # Create a temporary static core if someone wants to set predictor before first assess
            root = self._static_out_dir or "./runs/risk/adhoc"
            self._core = EpistemicGuard(root_out_dir=root, thresholds=self._thresholds, seed=self._seed)

        pr = getattr(predictor, "predict_risk", None)
        if pr is None:
            raise TypeError("predictor must have a `predict_risk` method")

        if asyncio.iscoroutinefunction(pr):
            self._core.set_predictor(predictor)
            return

        class _AsyncAdapter:
            async def predict_risk(self, question: str, context: str, **kw):
                return pr(question, context, **kw)

        self._core.set_predictor(_AsyncAdapter())


class EGVisualService(Service):
    """
    Thin pass-through for EGVisual; useful if you render custom figures.
    Prefer assess(run_id=...) above, which already renders field/strip/legend/badge.
    """

    def __init__(self):
        self._logger = logging.getLogger(self.name)
        self._visual: EGVisual | None = None
        self._up = False

    @property
    def name(self) -> str:
        return "eg-visual-v2"

    def initialize(self, **kwargs) -> None:
        cfg = (kwargs.get("config") or {}) if kwargs else {}
        logger = kwargs.get("logger")
        if logger is not None:
            self._logger = logger

        out_dir = cfg.get("out_dir", "./runs/risk/adhoc/img")
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
