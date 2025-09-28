# sis/routes/tap.py
from __future__ import annotations
from typing import Optional, List
import asyncio
import json
import time
import contextlib
import sys
import traceback
from fastapi import HTTPException

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

try:
    from nats.aio.client import Client as NATS
except Exception:
    NATS = None  # NATS optional

router = APIRouter(prefix="/tap", tags=["tap"])

# --- logging helper ----------------------------------------------------------
def log(*args):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[tap {ts}]", *args, file=sys.stdout, flush=True)


# --- tiny NATS helper (core only, no JetStream) -----------------------------
class _NatsCore:
    def __init__(self):
        self.nc: Optional[NATS] = None

    async def ensure(self, servers: List[str]) -> Optional[NATS]:
        log("NATS.ensure called; servers=", servers)
        if NATS is None:
            log("NATS lib not available")
            return None
        if self.nc and not self.nc.is_closed:
            log(
                "NATS.ensure: reusing existing nc (connected=",
                not self.nc.is_closed,
                ")",
            )
            return self.nc
        self.nc = NATS()
        log("NATS.ensure: connecting…")
        await self.nc.connect(
            servers=servers,
            name="sis-tap",
            allow_reconnect=True,
            reconnect_time_wait=2.0,
            max_reconnect_attempts=-1,
            ping_interval=10,
            max_outstanding_pings=5,
        )
        log("NATS.ensure: connected")
        return self.nc


_nats = _NatsCore()


def _nats_servers(request: Request):
    servers = getattr(request.app.state, "nats_servers", None) or [
        "nats://localhost:4222"
    ]
    log("nats servers from app.state:", servers)
    return servers


# --- DB page (list + detail) ------------------------------------------------
@router.get("", response_class=HTMLResponse)
def tap_page(request: Request):
    """
    DB-backed Arena view (list + detail from bus_events).
    """
    log("GET /tap page requested (DB mode)")
    templates = request.app.state.templates
    return templates.TemplateResponse("/arena/db.html", {"request": request})

# sis/routes/tap.py (add page)
@router.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("/arena/runs.html", {"request": request})

@router.get("/live", response_class=HTMLResponse)
def live_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("/arena/live.html", {"request": request})

# sis/routes/tap.py (only the changed/added bits shown)


def _store(request: Request):
    mem = getattr(request.app.state, "memory", None)
    if not mem or not getattr(mem, "bus_events", None):
        raise RuntimeError("app.state.memory.bus_events is not configured")
    return mem.bus_events

# --- API (DB-backed) --------------------------------------------------------

@router.get("/api/runs")
def api_recent_runs(request: Request, limit: int = Query(50, ge=1, le=500)):
    store = _store(request)
    try:
        return store.recent_runs(limit=limit)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/api/run/{run_id}/events")
def api_run_events(request: Request, run_id: str, limit: int = Query(2000, ge=1, le=10000)):
    store = _store(request)
    try:
        return store.payloads_by_run(run_id, limit=limit)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/api/recent")
def api_recent(
    request: Request,
    limit: int = Query(200, ge=1, le=2000),
    run_id: Optional[str] = Query(None),
    subject_like: Optional[str] = Query(None),
    event: Optional[str] = Query(None),
    since_id: Optional[int] = Query(None),
):
    store = _store(request)
    try:
        if since_id:
            rows = store.since_id(since_id, limit=limit)
        else:
            rows = store.recent(limit=limit, run_id=run_id, subject_like=subject_like, event=event)
        return [r.to_dict(include_payload=False, include_extras=False) for r in rows]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/api/event/{event_id}")
def api_event_detail(request: Request, event_id: int):
    store = _store(request)
    row = store.get(event_id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return row.to_dict(include_payload=True, include_extras=True)

# --- (Optional) Keep your SSE tap stream for debugging live traffic ----------
@router.get("/stream")
async def stream(
    request: Request,
    subject: str = Query(
        ">", description="NATS subject or wildcard (default '>')"
    ),
    debug: int = Query(0),
):
    log(f"GET /tap/stream open: subject={subject!r} debug={debug}")

    async def gen():
        try:
            nc = await _nats.ensure(_nats_servers(request))
        except Exception as e:
            err = {
                "event": "_error",
                "where": "connect",
                "type": type(e).__name__,
                "msg": str(e),
                "repr": repr(e),
                "trace": traceback.format_exc(),
                "t": time.time(),
            }
            log("ERROR connect:", json.dumps(err, ensure_ascii=False))
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            return

        queue: asyncio.Queue = asyncio.Queue()
        sub = None

        async def cb(msg):
            try:
                payload = msg.data.decode()
            except Exception:
                payload = ""
            out = {"subject": msg.subject, "data": None}
            try:
                out["data"] = json.loads(payload)
                log("NATS CB: subject=", msg.subject, " (JSON payload)")
            except Exception:
                out["data"] = payload
                log("NATS CB: subject=", msg.subject, " (RAW payload)")
            await queue.put(out)
            log("NATS CB: queued item for SSE")

        try:
            log("Subscribing to NATS subject:", subject)
            sub = await nc.subscribe(subject, cb=cb)
            log("Subscribed OK:", subject, " sub=", sub)
        except Exception as e:
            err = {
                "event": "_error",
                "where": "subscribe",
                "type": type(e).__name__,
                "msg": str(e),
                "repr": repr(e),
                "trace": traceback.format_exc(),
                "t": time.time(),
            }
            log("ERROR subscribe:", json.dumps(err, ensure_ascii=False))
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            return

        log("SSE: opening stream with initial comment and _open event")
        yield ":\n\n"
        yield f"data: {json.dumps({'event': '_open', 'subject': subject, 't': time.time()}, ensure_ascii=False)}\n\n"

        try:
            while True:
                try:
                    log("SSE loop: waiting for next item…")
                    item = await asyncio.wait_for(queue.get(), timeout=100.0)
                    log(
                        "SSE loop: got item, yielding to client; subject=",
                        item.get("subject"),
                    )
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    log("SSE loop: timeout; sending heartbeat")
                    yield ":\n\n"
                try:
                    if await request.is_disconnected():
                        log("SSE loop: client disconnected; breaking")
                        break
                except Exception as e:
                    log(
                        "SSE loop: request.is_disconnected check error:",
                        repr(e),
                    )
        except Exception as e:
            err = {
                "event": "_error",
                "where": "sse_loop",
                "type": type(e).__name__,
                "msg": str(e),
                "repr": repr(e),
                "trace": traceback.format_exc(),
                "t": time.time(),
            }
            log("ERROR sse_loop:", json.dumps(err, ensure_ascii=False))
            try:
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            except Exception:
                pass
        finally:
            log("SSE finally: cleanup")
            if sub:
                with contextlib.suppress(Exception):
                    log("SSE finally: unsubscribing…")
                    await sub.unsubscribe()
                    log("SSE finally: unsubscribed")

    resp = StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    log("GET /tap/stream returning StreamingResponse")
    return resp


@router.get("/api/events")
def api_events(request: Request, run_id: str):
    """
    All event bodies (payload_json) for a run; ascending time.
    """
    try:
        store = _store(request)
        bodies = store.payloads_by_run(run_id)
        return bodies
    except Exception as e:
        log("ERROR /tap/api/events:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


