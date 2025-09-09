# sis/routes/mind.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import asyncio
import json
import time

from fastapi import APIRouter, Request, Query, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

router = APIRouter(prefix="/mind", tags=["mind"])

# ---------- config helpers ----------

def _snap_dir(request: Request) -> Path:
    """
    Directory that holds snapshots. If app.state.viewer.snapshot_dir is set, prefers it.
    Fallback: ./runs/knowledge
    """
    viewer = getattr(request.app.state, "viewer", None)
    if viewer and getattr(viewer, "snapshot_dir", None):
        return Path(viewer.snapshot_dir)
    if isinstance(viewer, dict) and "snapshot_dir" in viewer:
        return Path(viewer["snapshot_dir"])
    return Path("./runs/knowledge")

def _snap_path(request: Request, snap_id: Optional[str]) -> Path:
    """
    Resolve a specific snapshot file:
      - if snap_id provided: <dir>/<snap_id>.json  (also accepts raw filename with .json)
      - else: <dir>/snapshot.json  (the live one)
      - if app.state.viewer.snapshot_path is set and snap_id is None → use it
    """
    viewer = getattr(request.app.state, "viewer", None)
    if snap_id:
        p = Path(snap_id)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        if not p.is_absolute():
            p = _snap_dir(request) / p.name
        return p
    # use configured live path if present
    if viewer and getattr(viewer, "snapshot_path", None):
        return Path(viewer.snapshot_path)
    if isinstance(viewer, dict) and "snapshot_path" in viewer:
        return Path(viewer["snapshot_path"])
    # default live file
    return _snap_dir(request) / "snapshot.json"

def _index_snapshots(request: Request, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Scan snapshot_dir for *.json and return quick metadata (sorted by mtime desc).
    Expected snapshot schema: {"nodes":[...], "edges":[...], "t": <timestamp>, ...}
    """
    d = _snap_dir(request)
    if not d.exists():
        return []
    items: List[Dict[str, Any]] = []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            stat = p.stat()
            items.append({
                "id": p.stem,          # snap_id
                "file": p.name,
                "path": str(p),
                "mtime": int(stat.st_mtime),
                "size": stat.st_size
            })
        except Exception:
            continue
        if len(items) >= limit:
            break
    return items

# ---------- pages ----------

@router.get("", response_class=HTMLResponse)
def list_page(
    request: Request,
    limit: int = Query(default=200, ge=1, le=1000),
):
    """
    /mind  → list of available snapshot files (newest first)
    """
    templates = request.app.state.templates
    snaps = _index_snapshots(request, limit=limit)
    return templates.TemplateResponse(
        "/mind/list.html",
        {
            "request": request,
            "snapshots": snaps,
            "snapshot_dir": str(_snap_dir(request)),
            "limit": limit,
            "now": int(time.time()),
        },
    )

@router.get("/view", response_class=HTMLResponse)
def detail_page(
    request: Request,
    id: Optional[str] = Query(default=None, description="Snapshot id or filename (.json optional). Omit for live snapshot."),
):
    """
    /mind/view?id=<snap_id>  → three.js viewer bound to a specific snapshot file (or live if omitted)
    """
    templates = request.app.state.templates
    snap_path = _snap_path(request, id)
    # Render even if file missing; the viewer shows empty graph until file appears
    return templates.TemplateResponse(
        "/mind/detail.html",
        {
            "request": request,
            "snap_id": id or "",
            "snap_path": str(snap_path),
        },
    )

# ---------- data / stream ----------

@router.get("/snapshot")
def snapshot_api(
    request: Request,
    id: Optional[str] = Query(default=None, description="Snapshot id or filename; omit for live snapshot"),
):
    """
    Return the snapshot JSON. Client viewer polls once and then subscribes to /mind/ws.
    """
    p = _snap_path(request, id)
    if not p.exists():
        # valid empty schema for the viewer
        return JSONResponse({"nodes": [], "edges": [], "t": 0})
    try:
        data = json.loads(p.read_text())
        data.setdefault("nodes", [])
        data.setdefault("edges", [])
        if "t" not in data:
            data["t"] = int(p.stat().st_mtime)
        return JSONResponse(data)
    except Exception as e:
        return PlainTextResponse(f"Malformed snapshot: {e}", status_code=500)

@router.websocket("/ws")
async def snapshot_ws(
    request: Request,
    ws: WebSocket,
    id: Optional[str] = None,
):
    """
    WebSocket that pushes the snapshot content whenever the file changes (by mtime/bytes).
    Client connects as:  ws://host/mind/ws?id=<snap_id>
    """
    await ws.accept()
    p = _snap_path(request, id)
    last_txt: Optional[str] = None
    try:
        while True:
            await asyncio.sleep(0.5)
            if not p.exists():
                continue
            txt = p.read_text()
            if txt != last_txt:
                last_txt = txt
                await ws.send_text(txt)
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
