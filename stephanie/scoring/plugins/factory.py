# stephanie/scoring/plugins/factory.py
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from .registry import get as get_registered


def _import_by_path(path: str):
    # Support "package.module:ClassName"
    if ":" in path:
        mod, cls = path.split(":", 1)
        m = importlib.import_module(mod)
        return getattr(m, cls)
    # Or "package.module.ClassName"
    parts = path.split(".")
    mod, cls = ".".join(parts[:-1]), parts[-1]
    m = importlib.import_module(mod)
    return getattr(m, cls)

def build_plugins(
    *,
    cfg: Dict[str, Any],
    container,
    logger,
    host_scorer,                # the scorer instance (optional for plugins that use host internals)
) -> List[Any]:
    """
    Reads cfg['plugins'] and instantiates enabled plugins.

    Accepted forms:
      plugins:
        scm:
          enabled: true
          class: "stephanie.scoring.plugins.scm_service_plugin:SCMServicePlugin"  # optional if registered
          params: { model_alias: "hf_hrm", topk: 0 }
        readability:
          enabled: false
          class: "stephanie.scoring.plugins.readability_plugin:ReadabilityPlugin"
          params: {}
    Or with registry names:
        scm:
          enabled: true
          params: { model_alias: "hf_hrm" }

    Each plugin ctor should accept: (container=None, logger=None, host=None, **params)
    """
    plugs_cfg: Dict[str, Any] = dict(cfg.get("plugins", {}))
    out: List[Any] = []
    for name, spec in plugs_cfg.items():
        if not isinstance(spec, dict):
            continue
        if not spec.get("enabled", False):
            continue

        cls = None
        if "class" in spec:
            cls = _import_by_path(str(spec["class"]))
        else:
            cls = get_registered(name)
            if cls is None:
                if logger:
                    logger.log("PluginMissingClass", {"plugin": name})
                continue

        params = dict(spec.get("params", {}))
        try:
            plugin = cls(container=container, logger=logger, host=host_scorer, **params)
            out.append(plugin)
        except Exception as e:
            if logger:
                logger.log("PluginInitError", {"plugin": name, "error": str(e)})
    return out
