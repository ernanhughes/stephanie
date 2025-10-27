from __future__ import annotations
import importlib
from typing import Any, Dict

def load_component_class(path: str):
    mod, _, cls = path.rpartition(".")
    if not mod:
        raise ImportError(f"Invalid component path: {path}")
    return getattr(importlib.import_module(mod), cls)

def build_component(spec: Dict[str, Any], **common_kwargs):
    cls_path = spec.get("cls")
    if not cls_path:
        raise ValueError("Component spec missing 'cls'")
    C = load_component_class(cls_path)
    args = dict(spec.get("args") or {})
    args.update(common_kwargs)
    return C(**args)
