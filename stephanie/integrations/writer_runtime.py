"""Bridge from Stephanie to the Writer runtime system.

Writer (https://github.com/.../Writer) exposes a large catalogue of
context-engineering runtimes.  Stephanie can *discover* them through the
Writer CLI (``writer info ... --json``) and, when the Writer packages are
importable, *execute* them in-process through their public facades.

This module keeps the two concerns separate:

* :class:`WriterRuntimeDiscovery` only shells out to the Writer CLI and
  never imports Writer.  It works from any machine that has the ``writer``
  executable on PATH (or a configured project root).
* :class:`WriterRuntimeExecutor` lazily imports ``writer_runtime`` and
  builds a ``Runtime`` gateway.  It is optional: if Writer is not
  installed/importable the discovery side still works and execution
  returns a clear ``unavailable`` result (never a silent fallback).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from stephanie.services.service_protocol import Service
except Exception:  # pragma: no cover
    from abc import ABC
    Service = ABC  # type: ignore

logger = logging.getLogger(__name__)

_STDERR_LIMIT = 2000


@dataclass
class WriterRuntimeConfig:
    project_root: Optional[str] = None
    writer_executable: Optional[str] = None
    python_executable: Optional[str] = None
    timeout_seconds: int = 30
    enable_execution: bool = False


class WriterRuntimeDiscovery:
    """Subprocess transport over the Writer CLI (discovery only)."""

    def __init__(self, cfg: WriterRuntimeConfig):
        self.cfg = cfg

    def _resolve_executable(self) -> str:
        if self.cfg.writer_executable:
            candidate = Path(self.cfg.writer_executable)
            if candidate.is_file():
                return str(candidate)
            raise FileNotFoundError(f"configured writer executable not found: {candidate}")
        on_path = shutil.which("writer")
        if on_path:
            return on_path
        if self.cfg.project_root:
            root = Path(self.cfg.project_root)
            for candidate in (root / "venv" / "Scripts" / "writer.exe",
                              root / "venv" / "Scripts" / "writer.cmd",
                              root / "venv" / "bin" / "writer"):
                if candidate.is_file():
                    return str(candidate)
        raise FileNotFoundError(
            "no Writer executable: set writer_executable, put `writer` on PATH, "
            "or set project_root to a checkout with venv/Scripts/writer"
        )

    def _run(self, args: List[str]) -> Dict[str, Any]:
        exe = self._resolve_executable()
        try:
            completed = subprocess.run(
                [exe, *args, "--json"],
                capture_output=True,
                text=True,
                timeout=self.cfg.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {"ok": False, "error": "timeout", "stderr": str(exc)[:_STDERR_LIMIT]}
        stderr = (completed.stderr or "")[:_STDERR_LIMIT]
        stdout = (completed.stdout or "").strip()
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": f"writer exited {completed.returncode}",
                "stderr": stderr,
                "stdout": stdout[:_STDERR_LIMIT],
            }
        try:
            payload = json.loads(stdout) if stdout else {}
        except json.JSONDecodeError:
            return {"ok": False, "error": "invalid JSON from writer", "stdout": stdout[:_STDERR_LIMIT], "stderr": stderr}
        if isinstance(payload, dict):
            payload.setdefault("ok", True)
            payload.setdefault("stderr", stderr)
            return payload
        return {"ok": True, "result": payload, "stderr": stderr}

    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        return self._run(["info", "search", query])

    def describe(self, runtime_name: str) -> Dict[str, Any]:
        return self._run(["info", "describe", runtime_name])

    def guide(self, runtime_name: str) -> Dict[str, Any]:
        return self._run(["info", "guide", runtime_name])

    def operations(self, runtime_name: str) -> Dict[str, Any]:
        return self._run(["info", "operations", runtime_name])

    def agent_context(self, runtime_name: str) -> Dict[str, Any]:
        return self._run(["info", "agent-context", runtime_name])


class WriterRuntimeExecutor:
    """Optional in-process execution against Writer facades."""

    def __init__(self, cfg: WriterRuntimeConfig):
        self.cfg = cfg
        self._runtime = None
        self._import_error: Optional[str] = None
        if cfg.enable_execution:
            self._prepare_import_path()

    def _prepare_import_path(self) -> None:
        if not self.cfg.project_root:
            self._import_error = "project_root not configured"
            return
        root = Path(self.cfg.project_root)
        candidates = [
            root / "packages" / name
            for name in (
                "writer-core",
                "writer-data",
                "writer-runtime",
                "writer-ai",
                "writer-external",
                "writer-film",
                "writer-api",
                "writer-cli",
                "writer-web",
                "opencode-writer",
            )
        ]
        # Also honour any other packages/<name> dirs added later.
        packages_dir = root / "packages"
        if packages_dir.is_dir():
            for child in sorted(packages_dir.iterdir()):
                if child.is_dir() and list(child.glob("*.egg-info")) :
                    candidates.append(child)
                elif child.is_dir() and any(g / "__init__.py" for g in child.iterdir() if g.is_dir()):
                    candidates.append(child)
        seen = set()
        for p in candidates:
            if p.is_dir() and str(p) not in seen and str(p) not in sys.path:
                sys.path.insert(0, str(p))
                seen.add(str(p))

    def available(self) -> bool:
        if self._import_error:
            return False
        try:
            import writer_runtime  # noqa: F401
            import writer_runtime.bootstrap  # noqa: F401
            return True
        except Exception as exc:  # pragma: no cover - environment dependent
            self._import_error = str(exc)
            return False

    def build_runtime(self) -> Dict[str, Any]:
        if not self.cfg.enable_execution:
            return {"ok": False, "error": "execution disabled by config"}
        if not self.available():
            return {"ok": False, "error": "writer runtime not importable", "detail": self._import_error}
        try:
            from writer_core.config import get_config
            from writer_runtime.bootstrap import build_runtime

            cfg = get_config()
            self._runtime = build_runtime(config=cfg)
            return {"ok": True, "message": "runtime built"}
        except Exception as exc:
            self._import_error = str(exc)
            return {"ok": False, "error": "failed to build runtime", "detail": str(exc)}

    def invoke(self, runtime_name: str, operation: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.cfg.enable_execution:
            return {"ok": False, "error": "execution disabled by config"}
        if self._runtime is None:
            built = self.build_runtime()
            if not built.get("ok"):
                return built
        if not self.available():
            return {"ok": False, "error": "writer runtime not importable", "detail": self._import_error}
        # Validate names like the OpenCode plugin does.
        import re
        if not re.match(r"^[a-z][a-z0-9_]{1,63}$", runtime_name or ""):
            return {"ok": False, "error": f"invalid runtime name: {runtime_name!r}"}
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]{1,63}$", operation or "") or operation.startswith("_"):
            return {"ok": False, "error": f"invalid operation name: {operation!r}"}
        try:
            facade = getattr(self._runtime, runtime_name, None)
            if facade is None:
                # Lazy materialization via attribute access may need a getter.
                getter = getattr(self._runtime, "get", None)
                if callable(getter):
                    facade = getter(runtime_name)
            if facade is None:
                return {"ok": False, "error": f"runtime not found: {runtime_name}"}
            method = getattr(facade, operation, None)
            if not callable(method):
                return {"ok": False, "error": f"operation not found: {runtime_name}.{operation}"}
            result = method(**(arguments or {}))
            payload = result
            model_dump = getattr(result, "model_dump", None)
            if callable(model_dump):
                payload = model_dump()
            return {"ok": True, "result": payload}
        except Exception as exc:
            return {"ok": False, "error": "invocation failed", "detail": str(exc)}


class WriterRuntimeBridge:
    """Convenience facade combining discovery + optional execution."""

    def __init__(self, cfg: WriterRuntimeConfig):
        self.cfg = cfg
        self.discovery = WriterRuntimeDiscovery(cfg)
        self.executor = WriterRuntimeExecutor(cfg)

    def status(self) -> Dict[str, Any]:
        return {
            "project_root": self.cfg.project_root,
            "execution_enabled": self.cfg.enable_execution,
            "writer_executable": self._safe_executable(),
            "runtime_importable": self.executor.available() if self.cfg.enable_execution else None,
        }

    def _safe_executable(self) -> Optional[str]:
        try:
            return self.discovery._resolve_executable()
        except Exception:
            return None


class WriterRuntimeService(Service):
    """Stephanie service wrapper around :class:`WriterRuntimeBridge`."""

    def __init__(self, cfg: Any = None, memory: Any = None, logger: Any = None, **kwargs: Any):
        self._cfg_raw = cfg
        self.memory = memory
        self.logger = logger or logging.getLogger(__name__)
        self._bridge: Optional[WriterRuntimeBridge] = None

    @property
    def name(self) -> str:  # type: ignore[override]
        return "writer-runtime-bridge"

    def initialize(self, **kwargs: Any) -> None:
        cfg = kwargs.get("config") or kwargs.get("cfg") or self._cfg_raw or {}
        if isinstance(cfg, dict) and "writer" in cfg and isinstance(cfg["writer"], dict):
            cfg = cfg["writer"]
        # Service container passes cfg as the whole hydra cfg sometimes.
        project_root = None
        writer_executable = None
        timeout_seconds = 30
        enable_execution = False
        if isinstance(cfg, dict):
            project_root = cfg.get("project_root")
            writer_executable = cfg.get("writer_executable")
            timeout_seconds = int(cfg.get("timeout_seconds", 30))
            enable_execution = bool(cfg.get("enable_execution", False))
        # Also allow nested under cfg.writer_runtime or args.
        if project_root is None and isinstance(self._cfg_raw, dict):
            nested = self._cfg_raw.get("writer_runtime", {}) if isinstance(self._cfg_raw.get("writer_runtime"), dict) else {}
            project_root = nested.get("project_root", project_root)
            writer_executable = nested.get("writer_executable", writer_executable)
            timeout_seconds = int(nested.get("timeout_seconds", timeout_seconds))
            enable_execution = bool(nested.get("enable_execution", enable_execution))
        wcfg = WriterRuntimeConfig(
            project_root=project_root,
            writer_executable=writer_executable,
            timeout_seconds=timeout_seconds,
            enable_execution=enable_execution,
        )
        self._bridge = WriterRuntimeBridge(wcfg)
        self.logger.info("WriterRuntimeService initialized (execution=%s)", enable_execution)

    def bridge(self) -> WriterRuntimeBridge:
        if self._bridge is None:
            self.initialize()
        assert self._bridge is not None
        return self._bridge

    def health_check(self) -> Dict[str, Any]:
        b = self._bridge
        if b is None:
            return {"status": "uninitialized", "timestamp": None}
        import datetime
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "details": b.status(),
        }

    def shutdown(self) -> None:
        self._bridge = None
