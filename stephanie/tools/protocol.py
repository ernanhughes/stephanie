# stephanie/tools/protocol.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

Json = Dict[str, Any]
ToolHandler = Callable[[Json], Awaitable[Json]]

@dataclass
class ToolSpec:
    name: str
    version: str = "1.0.0"
    summary: str = ""
    input_schema: Json = field(default_factory=dict)   # JSON Schema
    output_schema: Json = field(default_factory=dict)  # JSON Schema
    requires_services: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timeout_s: float = 15.0
    handler: Optional[ToolHandler] = None

# Decorator to register tools
_REGISTRY: List[ToolSpec] = []

def tool(spec: ToolSpec):
    def _wrap(fn: ToolHandler) -> ToolHandler:
        spec.handler = fn
        _REGISTRY.append(spec)
        return fn
    return _wrap

def registered_tools() -> List[ToolSpec]:
    return list(_REGISTRY)
