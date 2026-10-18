# sis/routes/logs.py
"""Log browsing routes.

Two bugs made /logs return 500:

* the routes read ``request.app.state.config``, but the app sets
  ``app.state.cfg`` (sis/main.py:68) -- ``AttributeError: 'State' object has no
  attribute 'config'``;
* the composed ``config/sis.yaml`` has no ``logs`` block at all (only the legacy
  ``sis/config/config.yaml`` does), so ``cfg["logs"]`` would have raised
  ``KeyError`` next.

Settings are now resolved defensively, with defaults, and file access is
confined to the configured directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter()

#: Used when the SIS config has no `logs` section.
DEFAULT_LOG_DIR = "logs"
DEFAULT_EXTENSIONS = (".log", ".jsonl", ".yaml")
DEFAULT_MAX_FILES = 500


def _log_settings(request: Request) -> Tuple[Path, Set[str], int]:
    """Resolve (log_dir, allowed_extensions, max_files) from the SIS config."""
    cfg = getattr(request.app.state, "cfg", None)

    section: Dict[str, Any] = {}
    if cfg is not None:
        try:
            section = dict(cfg.get("logs", {}) or {})
        except Exception:  # noqa: BLE001 - never 500 the log page over config
            logger.warning("logs: could not read the `logs` config section", exc_info=True)

    log_dir = Path(section.get("dir") or DEFAULT_LOG_DIR)
    extensions = section.get("extensions") or list(DEFAULT_EXTENSIONS)
    try:
        max_files = int(section.get("max_files") or DEFAULT_MAX_FILES)
    except (TypeError, ValueError):
        max_files = DEFAULT_MAX_FILES
    return log_dir, {str(e).lower() for e in extensions}, max_files


def _resolve_log_file(log_dir: Path, filename: str, allowed_exts: Set[str]) -> Path | None:
    """Return the log file, or None if it is outside log_dir / not allowed."""
    candidate = (log_dir / filename).resolve()
    try:
        candidate.relative_to(log_dir.resolve())
    except ValueError:
        logger.warning("logs: refused path outside the log dir: %r", filename)
        return None
    if not candidate.is_file():
        return None
    if allowed_exts and candidate.suffix.lower() not in allowed_exts:
        return None
    return candidate


@router.get("/logs", response_class=HTMLResponse)
def view_logs(request: Request):
    templates = request.app.state.templates
    log_dir, allowed_exts, max_files = _log_settings(request)

    files = []
    if log_dir.is_dir():
        for f in log_dir.iterdir():
            try:
                if not f.is_file():
                    continue
                if allowed_exts and f.suffix.lower() not in allowed_exts:
                    continue
                stat = f.stat()
                files.append(
                    {"name": f.name, "size": stat.st_size, "modified": stat.st_mtime}
                )
            except OSError:
                continue
    else:
        logger.warning("logs: directory not found: %s", log_dir)

    files = sorted(files, key=lambda x: x["modified"], reverse=True)[:max_files]

    return templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
            "files": files,
            "log_dir": str(log_dir),
            "active_page": "logs",
        },
    )


@router.get("/logs/view/{filename}", response_class=HTMLResponse)
def view_log_file(request: Request, filename: str):
    templates = request.app.state.templates
    log_dir, allowed_exts, _ = _log_settings(request)

    filepath = _resolve_log_file(log_dir, filename, allowed_exts)
    if filepath is None:
        return HTMLResponse(
            f"<h3>❌ File not allowed or not found: {filename}</h3>", status_code=404
        )

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    last_lines = lines[-500:]  # only last 500 lines
    return templates.TemplateResponse(
        "log_view.html",
        {
            "request": request,
            "filename": filename,
            "lines": last_lines,
            "active_page": "logs",
        },
    )


@router.get("/logs/download/{filename}")
def download_log_file(request: Request, filename: str):
    log_dir, allowed_exts, _ = _log_settings(request)

    filepath = _resolve_log_file(log_dir, filename, allowed_exts)
    if filepath is None:
        return HTMLResponse(
            f"<h3>❌ File not allowed or not found: {filename}</h3>", status_code=404
        )

    return FileResponse(filepath, filename=filename)
