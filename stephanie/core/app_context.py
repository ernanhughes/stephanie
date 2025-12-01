# -------------------------------
# FILE: stephanie/app/context.py
# -------------------------------
from __future__ import annotations

import importlib
from typing import Any, Optional

from omegaconf import OmegaConf

from stephanie.core.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.services.registry_loader import load_services_profile
from stephanie.services.service_container import ServiceContainer


class AppContext:
    """Single handle passed into agents/services.
    Replace (cfg, memory, container, logger) with this.
    """
    def __init__(
        self,
        cfg: Any,
        memory: MemoryTool,
        services: ServiceContainer,
        data_log: JSONLogger,
        run_name: str = "default_run",
    ):
        self.cfg = cfg
        self.memory = memory
        self.services = services
        self.data_log = data_log
        self.run_name = run_name

        profile_path = resolve_services_profile_path(cfg)
        load_services_profile(
            self.services,
            cfg=cfg,
            memory=self.memory,
            logger=self.data_log,
            profile_path=profile_path,
        )

    @classmethod
    def from_parts(
        cls,
        cfg: Any,
        memory: MemoryTool,
        services: ServiceContainer,
        data_log: JSONLogger,
        run_name: Optional[str] = None,
    ) -> AppContext:
        data_log = data_log or JSONLogger()
        return cls(
            cfg=cfg,
            memory=memory,
            services=services,
            data_log=data_log,
            run_name=run_name or "default_run",
        )

    @staticmethod
    def try_import(module: str, attr: Optional[str] = None) -> Any:
        """
        Import an optional dependency without raising if it is missing.

        Args:
            module: Dotted module path to import.
            attr: Optional attribute to fetch from the imported module.

        Returns:
            The imported module attribute, the module itself, or None if not found.
        """
        try:
            mod = importlib.import_module(module)
            return getattr(mod, attr) if attr else mod
        except (ImportError, AttributeError):
            return None


def resolve_services_profile_path(cfg: Any) -> str:
    prof = OmegaConf.select(cfg, "services.profile", default=None)
    if not prof:
        prof = "services/default"

    if prof.endswith(".yaml"):
        return prof
    return f"config/{prof}.yaml"
