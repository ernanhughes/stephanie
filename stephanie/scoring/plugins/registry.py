# stephanie/scoring/plugins/registry.py
from __future__ import annotations

from typing import Callable, Dict, Type

_REGISTRY: Dict[str, Type] = {}

def register(name: str) -> Callable[[Type], Type]:
    def _wrap(cls: Type) -> Type:
        _REGISTRY[name] = cls
        return cls
    return _wrap

def get(name: str):
    return _REGISTRY.get(name)
