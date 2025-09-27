# stephanie/services/arena_reporting_adapter.py
from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional


def _sf(v: Any):
    # JSON-safe numeric cast (handles numpy, float32, NaN/Inf)
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return float(v) if isinstance(v, (float,)) else v


class ArenaReporter:
    """Bridges Arena → ReportingService (and optionally EventService)."""

    def __init__(
        self,
        reporting_service,
        event_service=None,
        run_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.reporting = reporting_service
        self.events = event_service
        self.run_id = run_id or uuid.uuid4().hex
        self.meta = meta or {}

    async def start(self, ctx: Dict[str, Any], note="arena_start"):
        await self.reporting.emit(
            context={**ctx, "run_id": self.run_id},
            stage="arena",
            status="running",
            summary="Arena start",
            event="arena_start",
            **self.meta,
        )
        if self.events:
            await self.events.publish(
                "events.arena.run.start", {**self.meta, "run_id": self.run_id}
            )

    async def initial_scored(
        self, ctx: Dict[str, Any], scored_topk: List[Dict[str, Any]]
    ):
        topk = [
            {
                "origin": s.get("origin"),
                "variant": s.get("variant"),
                "overall": _sf((s.get("score") or {}).get("overall", 0.0)),
            }
            for s in scored_topk[: min(10, len(scored_topk))]
        ]
        payload = {**self.meta, "run_id": self.run_id, "topk": topk}
        await self.reporting.emit(
            context={**ctx, "run_id": self.run_id},
            stage="arena",
            event="initial_scored",
            **payload,
        )
        if self.events:
            await self.events.publish(
                "events.arena.run.initial_scored", payload
            )

    async def round_end(
        self,
        ctx: Dict[str, Any],
        round_ix: int,
        best_overall: float,
        marginal_per_ktok: float,
    ):
        payload = {
            **self.meta,
            "run_id": self.run_id,
            "round_ix": int(round_ix),
            "best_overall": _sf(best_overall),
            "marginal_per_ktok": _sf(marginal_per_ktok),
        }
        await self.reporting.emit(
            context={**ctx, "run_id": self.run_id},
            stage="arena",
            event="round_end",
            **payload,
        )
        if self.events:
            await self.events.publish("events.arena.run.round_end", payload)

    async def stop(
        self,
        ctx: Dict[str, Any],
        winner_overall: float,
        rounds_run: int,
        reason: str,
    ):
        payload = {
            **self.meta,
            "run_id": self.run_id,
            "winner_overall": _sf(winner_overall),
            "rounds_run": int(rounds_run),
            "reason": str(reason or "done"),
        }
        await self.reporting.emit(
            context={**ctx, "run_id": self.run_id},
            stage="arena",
            status="done",
            summary=f"winner={_sf(winner_overall):.3f} rounds={int(rounds_run)} reason={reason}",
            finalize=True,
            event="arena_stop",
            **payload,
        )
        if self.events:
            await self.events.publish("events.arena.run.stop", payload)
