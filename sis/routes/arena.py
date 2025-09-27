# sis/routes/arena.py
from __future__ import annotations
from operator import sub
from typing import Any, Dict, List, Optional
import asyncio
import json
import sqlite3
import time
import contextlib

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, PlainTextResponse

try:
    from nats.aio.client import Client as NATS
except Exception:
    NATS = None  # optional

# add near top of sis/routes/arena.py (if you haven’t already)
import asyncio
try:
    from anyio import get_cancelled_exc_class
    _Cancelled = get_cancelled_exc_class()  # AnyIO's cancel exception (BaseException)
except Exception:
    _Cancelled = asyncio.CancelledError

def _bus_stream(request: Request) -> str:
    return getattr(request.app.state, "bus_stream", "stephanie")  # default matches your bus


router = APIRouter(prefix="/arena", tags=["arena"])

# ---------- DB (optional persistence) ----------
def _db(request: Request) -> Optional[sqlite3.Connection]:
    dbpath = getattr(request.app.state, "arena_db_path", None)
    if not dbpath:
        return None
    conn = sqlite3.connect(dbpath, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS arena_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            event TEXT,
            payload TEXT,
            t REAL
        )
    """)
    return conn

def db_insert_event(conn: sqlite3.Connection, run_id: str, event: str, payload: Dict[str, Any], t: float):
    conn.execute("INSERT INTO arena_events (run_id,event,payload,t) VALUES (?,?,?,?)",
                 (run_id, event, json.dumps(payload, ensure_ascii=False), t))
    conn.commit()

def db_recent(conn: sqlite3.Connection, run_id: Optional[str], limit: int = 200) -> List[Dict[str, Any]]:
    q = "SELECT run_id, event, payload, t FROM arena_events "
    args: List[Any] = []
    if run_id:
        q += "WHERE run_id = ? "
        args.append(run_id)
    q += "ORDER BY id DESC LIMIT ?"
    args.append(limit)
    rows = conn.execute(q, args).fetchall()
    out: List[Dict[str, Any]] = []
    for r in reversed(rows):
        payload = {}
        try:
            payload = json.loads(r[2])
        except Exception:
            payload = {"raw": r[2]}
        payload.setdefault("event", r[1])
        payload.setdefault("run_id", r[0])
        payload.setdefault("t", r[3])
        out.append(payload)
    return out

# ---------- NATS helpers (live bridge) ----------
class _NatsHub:
    def __init__(self):
        self.nc: Optional[NATS] = None
        self.js = None

    async def ensure(self, servers: List[str]) -> Optional[NATS]:
        if NATS is None:
            return None
        if self.nc and not self.nc.is_closed:
            return self.nc
        self.nc = NATS()
        await self.nc.connect(
            servers=servers,
            name="sis-arena-bridge",
            allow_reconnect=True,
            reconnect_time_wait=2.0,
            max_reconnect_attempts=-1,
            ping_interval=10,
            max_outstanding_pings=5,
        )
        self.js = self.nc.jetstream()
        return self.nc

_nats = _NatsHub()

def _nats_servers(request: Request) -> List[str]:
    servers = getattr(request.app.state, "nats_servers", None)
    return servers or ["nats://localhost:4222"]

# ---------- Pages ----------
@router.get("", response_class=HTMLResponse)
def arena_page(request: Request):
    """
    /arena  → Live Arena viewer (run_id filter optional via ?run_id=abc)
    """
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "/arena/live.html",
        {"request": request}
    )

# ---------- API ----------
@router.get("/api/recent")
def api_recent(request: Request,
               run_id: Optional[str] = Query(None),
               limit: int = Query(200, ge=1, le=2000)):
    """
    Optional: Prefill UI with recent events (from SQLite) before SSE opens.
    """
    conn = _db(request)
    if not conn:
        return JSONResponse([])
    try:
        return JSONResponse(db_recent(conn, run_id, limit))
    finally:
        conn.close()

@router.get("/stream")
async def stream(
    request: Request,
    run_id: Optional[str] = Query(None),
    debug: int = Query(0),
):
    """
    SSE: relays NATS 'events.arena.run.>' (optional run_id filter).
    Fallbacks:
      - DB polling if NATS unavailable or subscribe fails
      - heartbeat only if neither NATS nor DB is available
    """

    def _sse(gen):
        return StreamingResponse(
            gen,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ---- Try NATS first
    nc = None
    try:
        nc = await _nats.ensure(_nats_servers(request))
    except Exception as e:
        if debug:
            print("[/arena/stream] NATS connect failed:", repr(e))
        nc = None

    if nc:
        queue: asyncio.Queue = asyncio.Queue()

        async def cb(msg):
            try:
                raw = json.loads(msg.data.decode())
            except Exception:
                return

            # Unwrap NATS bus envelope if present
            body = raw.get("payload") if isinstance(raw, dict) and "payload" in raw else raw
            print(body)
            if not isinstance(body, dict):
                return

            # Normalized run_id for optional filter
            rid = body.get("run_id") or body.get("arena_run_id") or body.get("runId")
            if run_id and rid != run_id:
                return

            await queue.put(body)
            print("[/arena/stream] enqueued", body.get("event"), "rid:", rid)

        subs = []
        subjects = [
            "events.arena.run.>",
            "stephanie.events.arena.run.>",
            f"{_bus_stream(request)}.events.arena.run.>",  # prefixed (e.g., stephanie.events...)
        ]
        for subj in subjects:
            try:
                s = await _nats.nc.subscribe(subj, cb=cb)
                subs.append(s)
                print("[/arena/stream] core subscribe:", subj)
            except Exception as e:
                print("[/arena/stream] core subscribe failed", subj, repr(e))

        if subs:
            async def nats_gen():
                yield ":\n\n"
                if debug:
                    yield 'data: {"event":"_debug","msg":"NATS connected"}\n\n'
                try:
                    while True:
                        try:
                            data = await asyncio.wait_for(queue.get(), timeout=5.0)
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                        except (asyncio.TimeoutError, TimeoutError):
                            try:
                                if await request.is_disconnected():
                                    break
                            except Exception:
                                pass
                            yield ":\n\n"  # heartbeat during idle
                            continue
                except _Cancelled:
                    # client disconnected — normal
                    pass
                finally:
                    for s in subs:
                        with contextlib.suppress(Exception):
                            await s.unsubscribe()
            return _sse(nats_gen())
        # If we got here, NATS is up but subscribe failed — fall through to DB

    # ---- DB polling fallback
    conn = _db(request)
    if conn:
        async def db_gen():
            yield ":\n\n"
            if debug:
                yield 'data: {"event":"_debug","msg":"DB polling"}\n\n'
            last_len = 0
            try:
                while True:
                    try:
                        if await request.is_disconnected():
                            break
                    except Exception:
                        pass
                    items = db_recent(conn, run_id, limit=500)
                    if len(items) > last_len:
                        for it in items[last_len:]:
                            yield f"data: {json.dumps(it, ensure_ascii=False)}\n\n"
                        last_len = len(items)
                    else:
                        yield ":\n\n"  # heartbeat
                    await asyncio.sleep(1.0)
            except _Cancelled:
                pass
            finally:
                conn.close()
        return _sse(db_gen())

    # ---- Last resort: heartbeat so curl/UI shows *something*
    async def hb_gen():
        yield ":\n\n"
        if debug:
            yield 'data: {"event":"_debug","msg":"heartbeat only (no NATS/DB)"}\n\n'
        i = 0
        try:
            while True:
                try:
                    if await request.is_disconnected():
                        break
                except Exception:
                    pass
                i += 1
                yield f'data: {json.dumps({"event":"_hb","i":i,"t":time.time()})}\n\n'
                try:
                    await asyncio.sleep(1.0)
                except _Cancelled:
                    break
        except _Cancelled:
            pass
    return _sse(hb_gen())
