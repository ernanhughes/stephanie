# stephanie/types/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


@dataclass
class ModelSpec:
    name: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_cfg(
        default_cfg: Dict[str, Any],
        override: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> ModelSpec:
        if override is None:
            base = default_cfg or {}
            m = base.get("model", {}) if "model" in base else base
            return ModelSpec(
                name=m.get("name") or "ollama/qwen:0.5b",
                api_base=m.get("api_base"),
                api_key=m.get("api_key"),
                params=(m.get("params") or {}),
            )
        if isinstance(override, str):
            return ModelSpec(name=override)
        return ModelSpec(
            name=override.get("name"),
            api_base=override.get("api_base"),
            api_key=override.get("api_key"),
            params=(override.get("params") or {}),
        )
