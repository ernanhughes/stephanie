# sis/routes/tap.py
from __future__ import annotations
from typing import Optional, List
import asyncio, json, time, contextlib, sys, traceback

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse

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
            log("NATS.ensure: reusing existing nc (connected=", not self.nc.is_closed, ")")
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
    servers = getattr(request.app.state, "nats_servers", None) or ["nats://localhost:4222"]
    log("nats servers from app.state:", servers)
    return servers

# --- page -------------------------------------------------------------------
@router.get("", response_class=HTMLResponse)
def tap_page(request: Request):
    """Minimal page that prints every SSE message."""
    log("GET /tap page requested")
    templates = request.app.state.templates
    return templates.TemplateResponse("/arena/live.html", {"request": request})

# --- SSE: subscribe and forward messages as-is ------------------------------
@router.get("/stream")
async def stream(
    request: Request,
    subject: str = Query(">", description="NATS subject or wildcard (default '>')"),
    debug: int = Query(0),
):
    """
    Opens a core NATS subscription to `subject` (wildcards ok) and forwards every
    message to the browser as Server-Sent Events. No filtering, no rewriting.
    """
    log(f"GET /tap/stream open: subject={subject!r} debug={debug}")

    async def gen():
        # Connect (or reuse) NATS
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
            # Try to parse JSON; if it fails, forward raw bytes as utf-8
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

        # Subscribe to the requested subject
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

        # open SSE stream
        log("SSE: opening stream with initial comment and _open event")
        yield ":\n\n"  # comment opens the stream
        open_evt = {"event": "_open", "subject": subject, "t": time.time()}
        yield f"data: {json.dumps(open_evt, ensure_ascii=False)}\n\n"

        try:
            while True:
                try:
                    log("SSE loop: waiting for next item…")
                    item = await asyncio.wait_for(queue.get(), timeout=100.0)
                    log("SSE loop: got item, yielding to client; subject=", item.get("subject"))
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    log("SSE loop: timeout; sending heartbeat")
                    yield ":\n\n"

                # stop if client disconnected
                try:
                    disconnected = await request.is_disconnected()
                    if disconnected:
                        log("SSE loop: client disconnected; breaking")
                        break
                except Exception as e:
                    log("SSE loop: request.is_disconnected check error:", repr(e))

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
            # Try to deliver the error to the client before closing
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
