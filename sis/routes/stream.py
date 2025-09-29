# sis/routes/stream.py
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

from nats.aio.client import Client as NATS

# --- logging helper ----------------------------------------------------------
def log(*args):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[tap {ts}]", *args, file=sys.stdout, flush=True)


def nats_servers(request: Request):
    servers = getattr(request.app.state, "nats_servers", None) or [
        "nats://localhost:4222"
    ]
    log("nats servers from app.state:", servers)
    return servers

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



# --- (Optional) Keep your SSE tap stream for debugging live traffic ----------
@router.get("/stream")
async def stream(
    request: Request,
    subject: str = Query(
        ">", description="NATS subject or wildcard (default '>')"
    ),
    debug: int = Query(0),
):
    print(f"GET /arena/stream open: subject={subject!r} debug={debug}")

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
            print("ERROR connect:", json.dumps(err, ensure_ascii=False))
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
                print("NATS CB: subject=", msg.subject, " (JSON payload)")
            except Exception:
                out["data"] = payload
                print("NATS CB: subject=", msg.subject, " (RAW payload)")
            await queue.put(out)
            print("NATS CB: queued item for SSE")

        try:
            print("Subscribing to NATS subject:", subject)
            sub = await nc.subscribe(subject, cb=cb)
            print("Subscribed OK:", subject, " sub=", sub)
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
            print("ERROR subscribe:", json.dumps(err, ensure_ascii=False))
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            return

        print("SSE: opening stream with initial comment and _open event")
        yield ":\n\n"
        yield f"data: {json.dumps({'event': '_open', 'subject': subject, 't': time.time()}, ensure_ascii=False)}\n\n"

        try:
            while True:
                try:
                    print("SSE loop: waiting for next item…")
                    item = await asyncio.wait_for(queue.get(), timeout=100.0)
                    print(
                        "SSE loop: got item, yielding to client; subject=",
                        item.get("subject"),
                    )
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    print("SSE loop: timeout; sending heartbeat")
                    yield ":\n\n"
                try:
                    if await request.is_disconnected():
                        print("SSE loop: client disconnected; breaking")
                        break
                except Exception as e:
                    print(
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
            print("ERROR sse_loop:", json.dumps(err, ensure_ascii=False))
            try:
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            except Exception:
                pass
        finally:
            print("SSE finally: cleanup")
            if sub:
                with contextlib.suppress(Exception):
                    print("SSE finally: unsubscribing…")
                    await sub.unsubscribe()
                    print("SSE finally: unsubscribed")

    resp = StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    print("GET /arena/stream returning StreamingResponse")
    return resp

