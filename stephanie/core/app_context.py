# -------------------------------
# FILE: stephanie/app/context.py
# -------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import OmegaConf

from stephanie.core.logging.json_logger import JSONLogger
from stephanie.core.manifest import Manifest
from stephanie.memory.memory_tool import MemoryTool
from stephanie.services.registry_loader import (_finalize_args, _load_object,
                                                _materialize_callables,
                                                _resolve_values)
from stephanie.services.service_container import ServiceContainer


@dataclass
class AppContext:
    """Single handle passed into agents/services.
    Replace (cfg, memory, container, logger) with this.
    """
    def __init__(self, cfg: Any, memory: MemoryTool, services: ServiceContainer, data_log: JSONLogger):
        self.cfg = cfg
        self.memory = memory    
        self.services = services
        self.data_log = data_log
        self.manifest = Manifest.start(cfg)

        profile_path = resolve_services_profile_path(cfg)
        load_services_profile(
            self.services,
            cfg=cfg,
            memory=self.memory,
            data_log=self.data_log,
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

def resolve_services_profile_path(cfg) -> str:
    prof = OmegaConf.select(cfg, "services.profile", default=None)
    if not prof:
        prof = "services/default"

    if prof.endswith(".yaml"):
        return prof
    return f"config/{prof}.yaml"


def load_services_profile(container: ServiceContainer, *, cfg, memory, data_log, profile_path: str, supervisor=None):
    raw = OmegaConf.load(profile_path)
    root = OmegaConf.to_container(raw, resolve=False)  # <-- prevents OmegaConf from resolving ${...}
    services_cfg = root.get("services", {})
    enabled = {n: s for n, s in services_cfg.items() if s.get("enabled", False)}

    # preflight: hard deps must be enabled
    for name, sc in enabled.items():
        for dep in (sc.get("dependencies") or []):
            if dep.startswith("?"): 
                continue
            if dep not in enabled:
                raise RuntimeError(f"Service '{name}' depends on '{dep}' which is disabled/missing in {profile_path}")

    for name, sc in enabled.items():
        deps = sc.get("dependencies") or []
        raw_args = sc.get("args") or {}         # constructor/factory args
        raw_init = sc.get("init_args") or {}    # optional initialize(**init_args)

        # Resolve placeholders
        resolved_args = _resolve_values(raw_args, container=container, cfg=cfg, memory=memory, logger=data_log)
        resolved_init = _resolve_values(raw_init, container=container, cfg=cfg, memory=memory, logger=data_log)

        # Bind supervisor + turn ${service:...} thunks into callables (late bind), then materialize
        finalized_args = _finalize_args(resolved_args, supervisor=supervisor)
        finalized_init = _finalize_args(resolved_init, supervisor=supervisor)

        def _materialize(obj):
            return _materialize_callables(obj)  # evaluates zero-arg callables (e.g., ${service:...})

        if "factory" in sc:
            factory = _load_object(sc["factory"])
            def provider(factory=factory, finalized=finalized_args):
                return factory(**_materialize(finalized))
            # init_args is only used if you explicitly provided init_args: in YAML
            init_kwargs = (lambda kw=finalized_init: _materialize(kw)) if raw_init else {}
            container.register(name=name, factory=provider, dependencies=deps, init_args=init_kwargs)

        elif "cls" in sc:
            cls = _load_object(sc["cls"])
            def provider(cls=cls, finalized=finalized_args):
                return cls(**_materialize(finalized))   # <-- pass args into __init__
            init_kwargs = (lambda kw=finalized_init: _materialize(kw)) if raw_init else {}
            container.register(name=name, factory=provider, dependencies=deps, init_args=init_kwargs)

        else:
            raise ValueError(f"Service '{name}' must define 'cls' or 'factory'")
