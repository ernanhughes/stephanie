from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse

router = APIRouter()

@router.get("/logs", response_class=HTMLResponse)
def view_logs(request: Request):
    templates = request.app.state.templates
    cfg = request.app.state.config
    log_dir = Path(cfg["logs"]["dir"])
    allowed_exts = set(cfg["logs"].get("extensions", []))  # e.g. [".log", ".jsonl", ".yaml"]
    max_files = cfg["logs"].get("max_files", 500)

    files = []
    for f in log_dir.iterdir():
        if f.is_file() and (not allowed_exts or f.suffix in allowed_exts):
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            })

    # Sort newest first, apply max_files limit
    files = sorted(files, key=lambda x: x["modified"], reverse=True)[:max_files]

    return templates.TemplateResponse(
        "logs.html",
        {"request": request, "files": files, "active_page": "logs"}
    )


@router.get("/logs/view/{filename}", response_class=HTMLResponse)
def view_log_file(request: Request, filename: str):
    templates = request.app.state.templates
    cfg = request.app.state.config
    log_dir = Path(cfg["logs"]["dir"])
    allowed_exts = set(cfg["logs"].get("extensions", []))

    filepath = log_dir / filename
    if not filepath.exists() or (allowed_exts and filepath.suffix not in allowed_exts):
        return HTMLResponse(f"<h3>❌ File not allowed or not found: {filename}</h3>", status_code=404)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    last_lines = lines[-500:]  # only last 500 lines
    return templates.TemplateResponse(
        "log_view.html",
        {"request": request, "filename": filename, "lines": last_lines, "active_page": "logs"}
    )


@router.get("/logs/download/{filename}")
def download_log_file(request: Request, filename: str):
    cfg = request.app.state.config
    log_dir = Path(cfg["logs"]["dir"])
    allowed_exts = set(cfg["logs"].get("extensions", []))

    filepath = log_dir / filename
    if not filepath.exists() or (allowed_exts and filepath.suffix not in allowed_exts):
        return HTMLResponse(f"<h3>❌ File not allowed or not found: {filename}</h3>", status_code=404)

    return FileResponse(filepath, filename=filename)
