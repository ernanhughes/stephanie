# stephanie/config/runconfig.py
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from stephanie.config.schema import AppConfig


class RunConfig(BaseModel):
    configurable: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"mcts.max_depth": 6}
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

def apply_overrides(cfg: AppConfig, rc: RunConfig) -> AppConfig:
    data = cfg.model_dump()
    for k, v in rc.configurable.items():
        cur = data
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    from stephanie.config.schema import AppConfig
    return AppConfig(**data)
