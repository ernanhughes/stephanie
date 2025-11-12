# stephanie/services/tool_registry_service.py
from __future__ import annotations


import importlib
import pkgutil
from typing import Any, Dict, List, Tuple

from stephanie.services.service_protocol import Service
from stephanie.tools.protocol import registered_tools, ToolSpec

class ToolRegistryService(Service):
    """
    Scans stephanie.tools.* for @tool-defined functions and exposes
    a runtime manifest filtered by service availability.
    """
    def __init__(self, packages: List[str] = None, service_facts: Dict[str, Any] = None):
        self._packages = packages or ["stephanie.tools"]
        self._service_facts = service_facts or {}
        self._manifest: Dict[str, Any] = {}
        self._initialized = False

    def initialize(self, **kwargs) -> None:
        # Load tool modules
        for pkg in self._packages:
            for m in pkgutil.walk_packages(importlib.import_module(pkg).__path__, pkg + "."):
                importlib.import_module(m.name)
        self._manifest = self._build_manifest()
        self._initialized = True

    def _is_service_up(self, name: str) -> bool:
        info = (self._service_facts.get(name) or {}).copy()
        # e.g., HybridKnowledgeBus health & backend
        # Expect shape like {"is_healthy": bool, "backend": "nats|inproc|none", ...}
        return bool(info.get("is_healthy") or info.get("status") in {"healthy"})

    def _enabled(self, spec: ToolSpec) -> Tuple[bool, Dict[str, Any]]:
        missing = [s for s in spec.requires_services if not self._is_service_up(s)]
        return (len(missing) == 0, {"missing_services": missing})

    def _build_manifest(self) -> Dict[str, Any]:
        tools = []
        for spec in registered_tools():
            ok, meta = self._enabled(spec)
            if not ok:
                continue
            tools.append({
                "name": spec.name,
                "version": spec.version,
                "summary": spec.summary,
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
                "tags": spec.tags,
                "timeout_s": spec.timeout_s,
            })
        return {"tools": tools, "services": self._service_facts}

    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy" if self._initialized else "unhealthy",
                "tool_count": len(self._manifest.get("tools", []))}

    def manifest(self) -> Dict[str, Any]:
        return dict(self._manifest)

    # Optional: direct invocation with guardrails
    async def call(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        from stephanie.tools.protocol import registered_tools
        mapping = {t.name: t for t in registered_tools()}
        spec = mapping.get(name)
        if not spec or not spec.handler:
            return {"error": f"Unknown tool '{name}'"}
        # You may validate params against spec.input_schema here.
        return await spec.handler(params)
