from __future__ import annotations
import os
import importlib
from typing import Any, Dict, Optional
from omegaconf import OmegaConf
from stephanie.services.service_container import ServiceContainer

class _DeferredServiceRef:
    def __init__(self, container: ServiceContainer, name: str, attr: Optional[str] = None):
        self.container, self.name, self.attr = container, name, attr
    def __call__(self):
        svc = self.container.get(self.name)
        return getattr(svc, self.attr) if self.attr else svc

def _load_object(dotted: str):
    # "pkg.mod:func" or "pkg.mod.Class"
    if ":" in dotted:
        mod, obj = dotted.split(":", 1)
        return getattr(importlib.import_module(mod), obj)
    mod, attr = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), attr)

def _select(cfg, path: str):
    try:
        return OmegaConf.select(cfg, path)
    except Exception:
        return None

def _resolve_values(v: Any, *, container: ServiceContainer, cfg, memory, logger):
    if isinstance(v, str):
        if v == "${cfg}": return cfg
        if v == "${memory}": return memory
        if v == "${logger}": return logger
        if v == "${container}": return container
        if v.startswith("${env:}"): return os.environ.get(v[7:-1], None)
        if v.startswith("${env:"):
            body = v[6:-1]
            if "|" in body:
                key, default = body.split("|", 1)
                return os.environ.get(key, default)
            return os.environ.get(body)
        if v.startswith("${cfg:"):
            return _select(cfg, v[6:-1])
        if v.startswith("${service:"):
            inner = v[10:-1]               # name or name.attr
            parts = inner.split(".", 1)
            name, attr = parts[0], (parts[1] if len(parts) == 2 else None)
            return _DeferredServiceRef(container, name, attr)
        if v.startswith("${supervisor:"):
            # replaced later in _finalize_args
            return ("__SUPERVISOR__", v[12:-1])
        return v
    if isinstance(v, dict):
        return {k: _resolve_values(val, container=container, cfg=cfg, memory=memory, logger=logger) for k, val in v.items()}
    if isinstance(v, list):
        return [_resolve_values(i, container=container, cfg=cfg, memory=memory, logger=logger) for i in v]
    return v

def _finalize_args(args: Dict[str, Any], supervisor=None):
    def fix(x):
        if isinstance(x, tuple) and len(x) == 2 and x[0] == "__SUPERVISOR__":
            if supervisor is None:
                raise RuntimeError("Supervisor method referenced but not provided")
            return getattr(supervisor, x[1])
        if isinstance(x, dict):  return {k: fix(v) for k, v in x.items()}
        if isinstance(x, list):  return [fix(i) for i in x]
        return x
    return fix(args)

def _materialize_callables(obj):
    if callable(obj):
        try: return obj()
        except TypeError: return obj
    if isinstance(obj, dict):
        return {k: _materialize_callables(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_materialize_callables(i) for i in obj]
    return obj

def load_services_profile(container: ServiceContainer, *, cfg, memory, logger, profile_path: str, supervisor=None):
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
        resolved_args = _resolve_values(raw_args, container=container, cfg=cfg, memory=memory, logger=logger)
        resolved_init = _resolve_values(raw_init, container=container, cfg=cfg, memory=memory, logger=logger)

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
