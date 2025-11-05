# stephanie/services/arena_reporting_adapter.py
from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional


def _sf(v: Any):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return float(v) if isinstance(v, float) else v

def _subject_for(event_name: str) -> str:
    mapping = {
        "arena_start": "start",
        "initial_scored": "initial_scored",
        "round_begin": "round_start",
        "round_start": "round_start",
        "round_end": "round_end",
        "arena_stop": "stop",
        "arena_done": "done",
    }
    return f"events.arena.run.{mapping.get(event_name, event_name)}"

def _prepare_emit(event_name: Optional[str], payload: Dict[str, Any]) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Returns (event_name, safe_for_reporting, safe_for_bus)

    - If payload contains 'event', prefer explicit event_name; otherwise adopt payload['event'].
    - Removes 'event' from kwargs passed to ReportingService.emit to avoid duplicate kw.
    - Keeps 'event' inside the bus payload.
    """
    body = dict(payload)  # shallow copy
    payload_event = body.pop("event", None)
    ev = event_name or payload_event or "unknown"
    bus_body = dict(body)
    bus_body["event"] = ev
    return ev, body, bus_body

class ArenaReporter:
    """Bridges Arena → ReportingService and (optionally) EventService.

    Supports BOTH styles:
      • new (arena-object): started(payload), round_start(payload), round_end(payload), done(payload)
      • legacy: start(ctx), initial_scored(ctx, topk), round_end_legacy(ctx,...), stop(ctx,...)
    """

    def __init__(self, reporting_service, event_service=None,
                 run_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        self.reporting = reporting_service
        self.events = event_service
        self.run_id = run_id
        if not self.run_id:
            self.run_id = f"{uuid.uuid4().hex}"
        self.meta = meta or {}

    # inside ArenaReporter
    async def __call__(self, payload: Dict[str, Any]):
        return await self.route(payload)

    async def route(self, payload: Dict[str, Any]):
        ev = (payload or {}).get("event")
        if ev == "arena_start":
            return await self.started(payload)
        if ev in ("round_begin", "round_start"):
            return await self.round_start(payload)
        if ev in ("round_end", "arena_stop"):
            return await self.round_end(payload)
        if ev == "arena_done":
            return await self.done(payload)
        # Unknown → still forward generically
        return await self._emit_generic(payload)

    async def _emit_generic(self, payload: Dict[str, Any]):
        ev = payload.get("event") or "arena_event"
        p = {"run_id": self.run_id, **self.meta, **payload, "event": ev}
        ev_name, rep_body, bus_body = _prepare_emit(ev, p)
        await self.reporting.emit(
            context={"run_id": p["run_id"], **self.meta},
            stage="arena",
            event=ev_name,
            **rep_body,
        )
        if self.events:
            await self.events.publish(_subject_for(ev_name), bus_body)


    # ---------- New arena-object style ----------
    async def started(self, payload: Dict[str, Any]):
        # payload: {"event":"arena_start", "run_id":..., "t":...}
        p = {"run_id": self.run_id, **self.meta, **payload}
        ev, rep_body, bus_body = _prepare_emit("arena_start", p)
        await self.reporting.emit(
            context={"run_id": p["run_id"], **self.meta},
            stage="arena",
            status="running",
            summary="Arena start",
            event=ev,
            **rep_body,
        )
        if self.events:
            await self.events.publish(_subject_for(ev), bus_body)

    async def round_start(self, payload: Dict[str, Any]):
        p = {"run_id": self.run_id, **self.meta, **payload}
        ev, rep_body, bus_body = _prepare_emit(p.get("event") or "round_start", p)
        await self.reporting.emit(
            context={"run_id": p["run_id"], **self.meta},
            stage="arena",
            event=ev,
            **rep_body,
        )
        if self.events:
            await self.events.publish(_subject_for(ev), bus_body)

    async def round_end(self, *args, **kwargs):
        """
        Flexible:
          • new style: round_end(payload_dict)
              payload keys: event="round_end", run_id, round, best_overall, marginal_per_ktok
          • legacy  : round_end(ctx, round_ix=..., best_overall=..., marginal_per_ktok=...)
        """
        # --- detect which style we're being called with ---
        if args and isinstance(args[0], dict) and (
            "event" in args[0] or "round" in args[0] or "best_overall" in args[0]
        ):
            payload = args[0]
            p = {"run_id": self.run_id, **self.meta, **payload}
            # normalize numerics
            if "best_overall" in p: p["best_overall"] = _sf(p["best_overall"])
            if "marginal_per_ktok" in p: p["marginal_per_ktok"] = _sf(p["marginal_per_ktok"])
            if "winner_overall" in p: p["winner_overall"] = _sf(p["winner_overall"])

            ev_hint = p.get("event") or "round_end"
            ev, rep_body, bus_body = _prepare_emit(ev_hint, p)

            await self.reporting.emit(
                context={"run_id": p["run_id"], **self.meta},
                stage="arena",
                status="running" if ev == "round_end" else "done",
                summary=(f"winner={p.get('winner_overall'):.3f} "
                         f"rounds={int(p.get('rounds_run',0))} reason={p.get('reason')}")
                         if ev == "arena_stop" else None,
                finalize=(ev == "arena_stop"),
                event=ev,
                **rep_body,
            )
            if self.events:
                await self.events.publish(_subject_for(ev), bus_body)
            return

        # legacy style: (ctx, round_ix=..., best_overall=..., marginal_per_ktok=...)
        ctx = args[0] if args else {}
        round_ix = int(kwargs.get("round_ix", 0))
        best_overall = _sf(kwargs.get("best_overall"))
        marginal_per_ktok = _sf(kwargs.get("marginal_per_ktok"))

        p = {
            "run_id": ctx.get("run_id") or self.run_id,
            **self.meta,
            "event": "round_end",
            "round_ix": round_ix,
            "round": round_ix,  # <-- add this for the UI
            "best_overall": best_overall,
            "marginal_per_ktok": marginal_per_ktok,
        }
        ev, rep_body, bus_body = _prepare_emit("round_end", p)

        await self.reporting.emit(
            context={**ctx, "run_id": p["run_id"]},
            stage="arena",
            status="running",
            event=ev,
            **rep_body,
        )
        if self.events:
            await self.events.publish(_subject_for(ev), bus_body)

    async def done(self, *args, **kwargs):
        # Accepts either done(payload_dict) OR done(ctx, ended_at=..., result=...)
        if args and isinstance(args[0], dict) and ("event" in args[0] or "ended_at" in args[0] or "summary" in args[0]):
            p = {"run_id": self.run_id, **self.meta, **args[0]}
        else:
            ctx = args[0] if args else {}
            p = {
                "run_id": ctx.get("run_id") or self.run_id,
                **self.meta,
                "event": "arena_done",
                "ended_at": kwargs.get("ended_at"),
                "result": kwargs.get("result"),
            }
        ev, rep_body, bus_body = _prepare_emit("arena_done", p)
        await self.reporting.emit(
            context={"run_id": p["run_id"], **self.meta},
            stage="arena",
            status="done",
            summary="Arena done",
            finalize=True,
            event=ev,
            **rep_body,
        )
        if self.events:
            await self.events.publish(_subject_for(ev), bus_body)

    # ---------- Legacy (ctx, ...) API ----------
    async def start(self, ctx: Dict[str, Any], note="arena_start"):
        p = {"run_id": self.run_id, **self.meta, "event": "arena_start", "note": note}
        ev, rep_body, bus_body = _prepare_emit("arena_start", p)
        await self.reporting.emit(
            context={**ctx, "run_id": self.run_id},
            stage="arena",
            status="running",
            summary="Arena start",
            event=ev,
            **rep_body,
        )
        if self.events:
            await self.events.publish(_subject_for(ev), bus_body)

    async def initial_scored(self, ctx: Dict[str, Any], scored_topk: List[Dict[str, Any]]):
        topk = [
            {"origin": s.get("origin"), "variant": s.get("variant"),
             "overall": _sf((s.get("score") or {}).get("overall", 0.0))}
            for s in scored_topk[: min(10, len(scored_topk))]
        ]
        p = {"run_id": self.run_id, **self.meta, "event": "initial_scored", "topk": topk}
        ev, rep_body, bus_body = _prepare_emit("initial_scored", p)
        await self.reporting.emit(
            context={**ctx, "run_id": self.run_id},
            stage="arena",
            event=ev,
            **rep_body,
        )
        if self.events:
            await self.events.publish(_subject_for(ev), bus_body)

    async def round_end_legacy(self, ctx: Dict[str, Any], round_ix: int, best_overall: float, marginal_per_ktok: float):
        p = {
            "run_id": self.run_id, **self.meta, "event": "round_end",
            "round_ix": int(round_ix),
            "best_overall": _sf(best_overall),
            "marginal_per_ktok": _sf(marginal_per_ktok),
        }
        ev, rep_body, bus_body = _prepare_emit("round_end", p)
        await self.reporting.emit(
            context={**ctx, "run_id": self.run_id},
            stage="arena",
            event=ev,
            **rep_body,
        )
        if self.events:
            await self.events.publish(_subject_for(ev), bus_body)

    async def stop(self, ctx: Dict[str, Any], winner_overall: float, rounds_run: int, reason: str):
        p = {
            "run_id": self.run_id, **self.meta, "event": "arena_stop",
            "winner_overall": _sf(winner_overall), "rounds_run": int(rounds_run), "reason": str(reason or "done"),
        }
        ev, rep_body, bus_body = _prepare_emit("arena_stop", p)
        await self.reporting.emit(
            context={**ctx, "run_id": self.run_id},
            stage="arena",
            status="done",
            summary=f"winner={_sf(winner_overall):.3f} rounds={int(rounds_run)} reason={reason}",
            finalize=True,
            event=ev,
            **rep_body,
        )
        if self.events:
            await self.events.publish(_subject_for(ev), bus_body)
