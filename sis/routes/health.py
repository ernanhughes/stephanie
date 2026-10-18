# sis/routes/health.py
"""Health / status routes.

* ``GET /healthz`` -- JSON, machine-readable. ``sis/security.py`` already treats
  ``/healthz`` as public, so this works for probes without a session.
* ``GET /health``  -- JSON alias; the local probes were already calling this.
* ``GET /status``  -- the human-readable page linked from the nav.

Every probe is defensive: a failing check degrades that row, it never turns the
endpoint into a 500.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter()

#: Checks that make the whole service unhealthy when they fail. Everything else
#: is informational (e.g. an intentionally disabled service is not a fault).
CRITICAL_CHECKS = {"database", "database_ready"}


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - health must never 500
        logger.warning("health probe failed: %s", exc, exc_info=True)
        return default


def _check(
    name: str,
    ok: bool,
    detail: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "status": "ok" if ok else "degraded",
        "critical": name in CRITICAL_CHECKS,
        "detail": detail,
        "data": data or {},
    }


def _cfg_section(cfg: Any, name: str) -> Dict[str, Any]:
    try:
        section = cfg.get(name, {}) if hasattr(cfg, "get") else {}
        return dict(section or {})
    except Exception:  # noqa: BLE001
        return {}


def _collect(request: Request) -> Dict[str, Any]:
    state = request.app.state
    checks: List[Dict[str, Any]] = []

    # --- database identity (the thing that was silently pointing at `co`) ----
    identity = _safe(lambda: state.memory.database_identity(), {}) or {}
    if not isinstance(identity, dict):
        identity = {"raw": str(identity)}
    matches = bool(identity.get("matches"))
    checks.append(
        _check(
            "database",
            matches,
            "connected database matches the configured one"
            if matches
            else "connected database does NOT match the configured one",
            identity,
        )
    )

    ready = bool(getattr(state, "database_ready", False))
    checks.append(
        _check(
            "database_ready",
            ready,
            f"boot identity check passed={ready}",
        )
    )

    # --- store counts ------------------------------------------------------
    counts: Dict[str, int] = {}
    for label in ("pipeline_runs", "papers", "casebooks", "cases", "documents"):
        store = getattr(state.memory, label, None)
        if store is None or not hasattr(store, "count"):
            continue
        value = _safe(lambda s=store: int(s.count()))
        if value is not None:
            counts[label] = value
    checks.append(
        _check("stores", bool(counts), f"{len(counts)} stores counted", counts)
    )

    # --- newest pipeline run (did anything actually run?) -------------------
    newest: Dict[str, Any] = {}
    rows = _safe(lambda: state.memory.pipeline_runs.list(limit=1), []) or []
    if rows:
        row = rows[0]
        newest = {
            "id": getattr(row, "id", None),
            "name": getattr(row, "name", None),
            "created_at": str(getattr(row, "created_at", "") or ""),
        }
    checks.append(
        _check(
            "pipeline_runs",
            bool(newest),
            f"newest run: {newest.get('id')} ({newest.get('created_at')})"
            if newest
            else "no pipeline runs recorded",
            newest,
        )
    )

    # --- Writer bridge -----------------------------------------------------
    bridge = getattr(state, "writer_bridge", None)
    wstatus = _safe(lambda: bridge.status(), {}) if bridge is not None else {}
    if not isinstance(wstatus, dict):
        wstatus = {}
    writer_ok = bool(wstatus.get("writer_executable"))
    checks.append(
        _check(
            "writer",
            writer_ok,
            "Writer CLI resolved" if writer_ok else "Writer CLI not found",
            wstatus,
        )
    )

    # --- optional services (informational) ---------------------------------
    cfg = getattr(state, "cfg", None)
    services = {}
    for name in ("worker", "chat_runtime", "experience_learning", "stn06_experiment", "writer"):
        section = _cfg_section(cfg, name)
        if section:
            services[name] = bool(section.get("enabled", False))
    checks.append(
        _check(
            "services",
            True,
            ", ".join(f"{k}={'on' if v else 'off'}" for k, v in sorted(services.items()))
            or "no optional services configured",
            services,
        )
    )

    critical_failures = [c["name"] for c in checks if c["critical"] and c["status"] != "ok"]
    if critical_failures:
        status = "unhealthy"
    elif any(c["status"] != "ok" for c in checks):
        status = "degraded"
    else:
        status = "healthy"

    return {
        "status": status,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "critical_failures": critical_failures,
        "checks": checks,
    }


@router.get("/healthz")
def healthz(request: Request) -> Dict[str, Any]:
    """Machine-readable health (public -- see sis/security.py)."""
    return _collect(request)


@router.get("/health")
def health(request: Request) -> Dict[str, Any]:
    """Alias for /healthz."""
    return _collect(request)


@router.get("/status", response_class=HTMLResponse)
def status_page(request: Request):
    """Human-readable health page."""
    templates = request.app.state.templates
    payload = _collect(request)
    return templates.TemplateResponse(
        "health.html",
        {
            "request": request,
            "payload": payload,
            "checks": payload["checks"],
            "active_page": "status",
        },
    )
