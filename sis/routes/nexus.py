# sis/routes/nexus.py
from __future__ import annotations
from fastapi import APIRouter, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import datetime
import re

from stephanie.utils.json_sanitize import sanitize

# Single router for everything Nexus
router = APIRouter(tags=["nexus"])

# Root where Nexus artifacts are written (override via config if you want)
BASE_DIR = Path("./runs/nexus_vpm").resolve()

# ----------------- helpers -----------------

def _safe_run_dir(run_id: str) -> Path:
    """Return the resolved path for a run id, preventing directory traversal."""
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]+", run_id or ""):
        raise HTTPException(status_code=400, detail="Invalid run id")
    p = (BASE_DIR / run_id).resolve()
    if not str(p).startswith(str(BASE_DIR)):
        raise HTTPException(status_code=400, detail="Invalid path resolution")
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return p

def _find_json(dirpath: Path, *candidates: str) -> Optional[Path]:
    """Try a few common names, otherwise return None."""
    for name in candidates:
        p = dirpath / name
        if p.exists():
            return p
    for p in dirpath.glob("*.json"):
        n = p.name.lower()
        if any(c in n for c in candidates):
            return p
    return None

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _pretty_time(ts: float) -> str:
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _get_nexus(app):
    comp = getattr(app.state, "nexus", None)
    if comp is None:
        raise RuntimeError("Nexus component not initialized")
    return comp

# ----------------- in-memory “live” state (optional) -----------------

_GRAPH: Dict[str, Any] = {"nodes": [], "edges": []}
_MANIFEST: Dict[str, Any] = {}
_STATUS: Dict[str, Any] = {"status": "idle"}

def set_graph(nodes: Dict[str, Any], edges: List[Dict[str, Any]]):
    """Convert your NexusNode/NexusEdge into Cytoscape elements."""
    _GRAPH["nodes"] = [
        {
            "data": {
                "id": nid,
                "label": getattr(n, "title", None) or getattr(n, "text", "")[:80] or nid,
                "type": getattr(n, "target_type", "unknown"),
                "deg": getattr(n, "degree", 0),
            }
        }
        for nid, n in nodes.items()
    ]
    _GRAPH["edges"] = [
        {
            "data": {
                "id": f"{e.src}->{e.dst}",
                "source": e.src,
                "target": e.dst,
                "type": e.type,
                "weight": float(getattr(e, "weight", 0.0) or 0.0),
            }
        }
        for e in edges
    ]

def set_manifest(manifest: Dict[str, Any]):
    _MANIFEST.clear()
    _MANIFEST.update(manifest or {})

def set_status(status: Dict[str, Any]):
    _STATUS.update(status or {})

# ----------------- HTML: /nexus = runs index -----------------

@router.get("/nexus", response_class=HTMLResponse)
def runs_index(request: Request):
    """
    HTML index listing all runs under BASE_DIR.
    """
    if not BASE_DIR.exists():
        BASE_DIR.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    for d in sorted([p for p in BASE_DIR.iterdir() if p.is_dir()],
                    key=lambda x: x.stat().st_mtime, reverse=True):
        manifest = None
        mpath = _find_json(d, "manifest.json", "nexus_manifest.json", "meta.json")
        if mpath:
            manifest = _load_json(mpath)
        runs.append({
            "id": d.name,
            "mtime": _pretty_time(d.stat().st_mtime),
            "manifest": {
                "counts": (manifest or {}).get("counts"),
                "run_id": (manifest or {}).get("run_id"),
            } if manifest else None,
        })
    return request.app.state.templates.TemplateResponse(
        "nexus/index.html",
        {"request": request, "runs": runs, "base_dir": str(BASE_DIR)},
    )

# (Optional) keep a “live dashboard” separate if you still want it
@router.get("/nexus/dashboard", response_class=HTMLResponse)
def nexus_dashboard(request: Request):
    return request.app.state.templates.TemplateResponse(
        "nexus/dashboard.html",
        {"request": request},
    )

# ----------------- live JSON endpoints -----------------

@router.get("/nexus/status", response_class=JSONResponse)
def status(request: Request):
    try:
        try:
            nx = _get_nexus(request.app)
            live = nx.status() if hasattr(nx, "status") else {}
        except Exception:
            live = {}
        out = {**_STATUS, **(live or {})}
        return JSONResponse(content=sanitize(out))
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/nexus/graph.json", response_class=JSONResponse)
def graph_json():
    return JSONResponse(content=sanitize(_GRAPH))

@router.get("/nexus/manifest", response_class=JSONResponse)
def manifest_json():
    return JSONResponse(content=sanitize(_MANIFEST or {}))

# ----------------- control endpoints -----------------

@router.post("/nexus/build", response_class=JSONResponse)
def build(
    request: Request,
    limit: int | None = Query(default=None, ge=1),
    knn_k: int = Query(default=12, ge=1),
    edge_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
    temporal: bool = Query(default=True),
):
    """
    One-shot rebuild of the Nexus graph from current loader/scorables.
    """
    nx = _get_nexus(request.app)
    nodes, edges, manifest = nx.build_graph(
        limit=limit,
        knn_k=knn_k,
        sim_threshold=edge_threshold,
        add_temporal=temporal,
    )
    set_graph(nodes, edges)
    from pathlib import Path
    run_id = (manifest or {}).get("run_id") or "unknown"
    run_dir = Path((manifest or {}).get("run_dir") or (manifest or {}).get("output_dir") or (BASE_DIR / run_id))
    run_dir.mkdir(parents=True, exist_ok=True)

    graph_payload = {
        "nodes": _GRAPH.get("nodes", []),
        "edges": _GRAPH.get("edges", []),
    }
    (run_dir / "graph.json").write_text(json.dumps(graph_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    set_manifest(manifest)
    set_status({"status": "built", "nodes": len(nodes), "edges": len(edges)})
    return JSONResponse(content=sanitize({
        "ok": True,
        "nodes": len(nodes),
        "edges": len(edges),
        "edge_types": {
            "knn_global": sum(1 for e in edges if e.type == "knn_global"),
            "temporal_next": sum(1 for e in edges if e.type == "temporal_next"),
        },
        "manifest": manifest,
    }))

@router.post("/nexus/finalize", response_class=JSONResponse)
async def finalize(request: Request):
    """
    Finalize any pending VPM timelines and refresh manifest/graph paths.
    """
    nx = _get_nexus(request.app)
    meta = await nx.finalize_current_run() if hasattr(nx, "finalize_current_run") else {}
    if "manifest" in meta:
        set_manifest(meta["manifest"])
    set_status({"status": "finalized"})
    return JSONResponse(content=sanitize(meta or {"ok": True}))

@router.get("/nexus/ping")
def ping():
    return Response(content="pong", media_type="text/plain")

# ----------------- per-run pages & data -----------------

@router.get("/nexus/run/{run_id}", response_class=HTMLResponse)
def run_view(request: Request, run_id: str):
    """
    HTML page for a single run with a Cytoscape graph + manifest panel.
    """
    _safe_run_dir(run_id)
    return request.app.state.templates.TemplateResponse(
        "nexus/run_view.html",
        {"request": request, "run_id": run_id},
    )

@router.get("/nexus/runs.json", response_class=JSONResponse)
def runs_json():
    if not BASE_DIR.exists():
        BASE_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for d in sorted([p for p in BASE_DIR.iterdir() if p.is_dir()],
                    key=lambda x: x.stat().st_mtime, reverse=True):
        m = _find_json(d, "manifest.json", "nexus_manifest.json", "meta.json")
        j = _load_json(m) if m else None
        runs.append({
            "id": d.name,
            "path": str(d),
            "mtime": d.stat().st_mtime,
            "mtime_human": _pretty_time(d.stat().st_mtime),
            "manifest": j,
        })
    return JSONResponse(runs)

@router.get("/nexus/run/{run_id}/manifest", response_class=JSONResponse)
def run_manifest(run_id: str):
    d = _safe_run_dir(run_id)
    p = _find_json(d, "nexus_manifest.json", "manifest.json", "meta.json")
    if not p:
        raise HTTPException(status_code=404, detail="manifest not found")
    data = _load_json(p) or {}
    return JSONResponse(data)

@router.get("/nexus/run/{run_id}/graph.json", response_class=JSONResponse)
def run_graph_json(run_id: str):
    """
    Serve saved graph elements if present.
    Expect file like graph.json with Cytoscape elements:
      { "nodes": [...], "edges": [...] }
    """
    d = _safe_run_dir(run_id)
    p = _find_json(d, "graph.json", "nexus_graph.json")
    if not p:
        return JSONResponse({"nodes": [], "edges": []})

    data = _load_json(p) or {}
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    resp = JSONResponse({"nodes": nodes, "edges": edges})
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp

@router.get("/nexus/run/{run_id}/artifact/{rel_path:path}")
def run_artifact(run_id: str, rel_path: str):
    """
    Serve an artifact file (png/gif/json) inside the run directory.
    """
    d = _safe_run_dir(run_id)
    target = (d / rel_path).resolve()
    if not str(target).startswith(str(d)):
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(target)

@router.get("/nexus/timeline", response_class=HTMLResponse)
def timeline_page(request: Request, baseline: str | None = None, targeted: str | None = None):
    return request.app.state.templates.TemplateResponse(
        "nexus/timeline.html",
        {"request": request, "baseline": baseline or "", "targeted": targeted or ""},
    )
