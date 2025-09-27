# sis/routes/arena.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import asyncio
import json
import time
import sys
import contextlib

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse

try:
    from nats.aio.client import Client as NATS
except Exception:
    NATS = None  # optional

# ---------- tiny logging helper ----------
def log(*args):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[arena {ts}]", *args, file=sys.stdout, flush=True)

# ---------- cancel type for AnyIO/asyncio ----------
try:
    from anyio import get_cancelled_exc_class
    _Cancelled = get_cancelled_exc_class()
except Exception:
    _Cancelled = asyncio.CancelledError

def _bus_stream(request: Request) -> str:
    # used to build a third subject: "<bus>.events.arena.run.>"
    return getattr(request.app.state, "bus_stream", "stephanie")

router = APIRouter(prefix="/arena", tags=["arena"])

# ---------- NATS helper (core-only) ----------
class _NatsCore:
    def __init__(self):
        self.nc: Optional[NATS] = None

    async def ensure(self, servers: List[str]) -> Optional[NATS]:
        log("NATS.ensure called; servers=", servers)
        if NATS is None:
            log("NATS lib not available (pip install nats-py)")
            return None
        if self.nc and not self.nc.is_closed:
            log("NATS.ensure: reusing existing connection")
            return self.nc
        self.nc = NATS()
        log("NATS.ensure: connecting…")
        await self.nc.connect(
            name="sis-arena",
            allow_reconnect=True,
            reconnect_time_wait=2.0,
            max_reconnect_attempts=-1,
            ping_interval=10,
            max_outstanding_pings=5,
        )
        log("NATS.ensure: connected")
        return self.nc

_nats = _NatsCore()

def _nats_servers(request: Request) -> List[str]:
    servers = getattr(request.app.state, "nats_servers", None) or ["nats://localhost:4222"]
    log("nats servers from app.state:", servers)
    return servers

# ---------- Page ----------
@router.get("", response_class=HTMLResponse)
def arena_page(request: Request):
    """
    /arena → keep your existing viewer template if you want.
    This route file focuses on the NATS-only /stream below.
    """
    templates = request.app.state.templates
    return templates.TemplateResponse("/arena/live.html", {"request": request})

# ---------- SSE (NATS-only) ----------
@router.get("/stream")
async def stream(
    request: Request,
    run_id: Optional[str] = Query(None, description="optional run_id filter; if present we drop non-matching messages *when run_id exists on the message*"),
    debug: int = 2
):
    """
    Subscribe to NATS subjects and forward messages as SSE.
    - NATS ONLY (no DB/heartbeat fallbacks)
    - Robust payload unwrap (payload.payload…)
    - Infers `event` from subject if missing
    - Verbose logging
    """

    async def gen():
        # 1) Connect to NATS
        try:
            nc = await _nats.ensure(_nats_servers(request))
        except Exception as e:
            err = {"event": "_error", "where": "connect", "msg": f"{type(e).__name__}: {e!s}"}
            log("CONNECT ERROR:", err)
            yield f"data: {json.dumps(err)}\n\n"
            return

        # 2) Build subjects and subscribe
        subs = []
        q: asyncio.Queue = asyncio.Queue()

        async def cb(msg):
            # Raw decode
            try:
                raw_s = msg.data.decode()
            except Exception as de:
                log("CB decode error:", repr(de))
                return

            # Try parse JSON
            obj = None
            try:
                obj = json.loads(raw_s)
                if debug:
                    log("CB parsed JSON for subject:", msg.subject)
            except Exception:
                if debug:
                    log("CB non-JSON payload; forwarding raw string for subject:", msg.subject)
                obj = raw_s  # keep as raw string

            # Unwrap envelope(s) until the deepest dict/list
            body = obj
            # common pattern: { ... "payload": {...} }
            # keep unwrapping while there is a "payload" (or "data") which is dict/list
            if isinstance(body, dict):
                changed = True
                while changed:
                    changed = False
                    if "payload" in body and isinstance(body["payload"], (dict, list)):
                        body = body["payload"]
                        changed = True
                    elif "data" in body and isinstance(body["data"], (dict, list)):
                        body = body["data"]
                        changed = True

            # At this point, "body" can be dict (ideal), list (we'll wrap each item), or string (raw).
            # Normalize into a list of items to forward (often a single dict).
            items: List[Any] = []
            if isinstance(body, list):
                items = body
            else:
                items = [body]

            # Apply optional run_id filter, but ONLY when the item has a run_id.
            for item in items:
                deliver = True
                rid = None
                if isinstance(item, dict):
                    rid = (
                        item.get("run_id")
                        or item.get("arena_run_id")
                        or item.get("runId")
                        or (item.get("meta") or {}).get("run_id")
                    )
                    # infer/attach event from subject if missing and subject looks like "...round_start"
                    if "event" not in item and isinstance(msg.subject, str):
                        try:
                            ev_hint = msg.subject.split(".")[-1]
                            # normalize round_begin → round_start to be consistent with your UI
                            if ev_hint == "round_begin":
                                ev_hint = "round_start"
                            item["event"] = ev_hint
                        except Exception:
                            pass

                if run_id and rid is not None and str(rid) != str(run_id):
                    deliver = False
                    if debug:
                        log(f"FILTER: drop message (rid={rid}) != filter({run_id}); subject={msg.subject}")

                if deliver:
                    out = {"subject": msg.subject, "data": item}
                    if debug:
                        log("ENQUEUE:", json.dumps(out)[:500] + ("..." if len(json.dumps(out)) > 500 else ""))
                    await q.put(out)

        subjects = [
            "events.arena.run.>",
            "stephanie.events.arena.run.>",
            f"{_bus_stream(request)}.events.arena.run.>",  # e.g. stephanie.events...
        ]
        # Use a set to avoid accidental duplication if bus_stream already equals "stephanie"
        _seen = set()
        subjects = [s for s in subjects if not (s in _seen or _seen.add(s))]

        for subj in subjects:
            try:
                s = await nc.subscribe(subj, cb=cb)
                subs.append(s)
                log("SUBSCRIBED:", subj)
            except Exception as e:
                log("SUBSCRIBE ERROR:", subj, repr(e))

        if not subs:
            err = {"event": "_error", "where": "subscribe", "msg": "no subscriptions established"}
            log("FATAL:", err)
            yield f"data: {json.dumps(err)}\n\n"
            return

        # 3) Open SSE
        yield ":\n\n"
        open_evt = {"event": "_open", "url": str(request.url), "t": time.time()}
        log("SSE OPEN:", open_evt)
        yield f"data: {json.dumps(open_evt)}\n\n"

        # 4) Initial hello (so onmessage fires ASAP)
        hello = {"event": "_hello", "t": time.time()}
        yield f"data: {json.dumps(hello)}\n\n"

        # 5) Pump queue → SSE
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=5.0)
                    # forward exactly what we captured
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                    if debug:
                        log("SSE SEND OK subject=", item.get("subject"))
                except asyncio.TimeoutError:
                    # keep alive heartbeat (comment)
                    yield ":\n\n"
                    if debug:
                        log("SSE heartbeat sent")

                # stop if client disconnected
                try:
                    if await request.is_disconnected():
                        log("Client disconnected, stopping SSE loop")
                        break
                except Exception as e:
                    log("is_disconnected() check error:", repr(e))
        except _Cancelled:
            log("Generator cancelled (normal shutdown)")
        except Exception as e:
            err = {"event": "_error", "where": "loop", "msg": f"{type(e).__name__}: {e!s}"}
            log("LOOP ERROR:", err)
            # best effort to notify client
            try:
                yield f"data: {json.dumps(err)}\n\n"
            except Exception:
                pass
        finally:
            # cleanup
            for s in subs:
                with contextlib.suppress(Exception):
                    await s.unsubscribe()
            log("Unsubscribed all; SSE closing")

    # Build response
    resp = StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    log("Returning StreamingResponse for /arena/stream (NATS-only)")
    return resp
