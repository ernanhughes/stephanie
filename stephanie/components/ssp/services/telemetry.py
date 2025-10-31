# stephanie/components/ssp/services/telemetry.py
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from stephanie.utils.date_utils import iso_now



class SSPTelemetry:
    """
    Unified event emitter for SSP:
      - Publishes to bus (memory.bus.publish)
      - Optionally persists to DB (memory.bus_events.insert or .insert_event)

    Config (Hydra):
      ssp:
        telemetry:
          enabled: true
          persist: true
          sample: 1.0
          subject_root: "ssp"
          extra: {}   # merged into every envelope
    """

    def __init__(self, cfg, memory, logger, subject_root: str = "ssp"):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        tcfg = (cfg.get("telemetry") or {}) if isinstance(cfg, dict) else {}
        self.enabled: bool = bool(tcfg.get("enabled", True))
        self.persist_default: bool = bool(tcfg.get("persist", True))
        self.sample: float = float(tcfg.get("sample", 1.0))
        self.subject_root: str = str(tcfg.get("subject_root", subject_root))
        self.extra: Dict[str, Any] = dict(tcfg.get("extra", {}) or {})

    # -------------------------
    # Core
    # -------------------------
    def envelope(
        self,
        subject: str,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = context or {}
        goal_obj = ctx.get("goal") if isinstance(ctx.get("goal"), dict) else None
        env = {
            "event_id": uuid.uuid4().hex,
            "ts": time.time(),
            "subject": subject,
            "pipeline_run_id": ctx.get("pipeline_run_id"),
            "ssp_run_id": ctx.get("ssp_run_id"),
            "tick_id": ctx.get("tick_id"),
            "goal_id": (goal_obj or {}).get("id") if goal_obj else ctx.get("goal_id"),
            "actor": ctx.get("actor"),
            **self.extra,
            "payload": payload,
        }
        return env

    async def publish(
        self,
        subject: str,
        payload: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        persist: Optional[bool] = None,
    ) -> None:
        if not self.enabled:
            return
        env = self.envelope(subject, payload, context)

        # 1) Bus
        try:
            bus = getattr(self.memory, "bus", None)
            if bus:
                # Publish *envelope* (subject already included inside)
                await bus.publish(subject, env)
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "SSPBusPublishError", {"subject": subject, "error": str(e)}
                )

        # 2) Optional DB persist
        try:
            do_persist = self.persist_default if persist is None else persist
            store = getattr(self.memory, "bus_events", None)
            if do_persist and store:
                # Support either insert(subject, envelope) or insert_event(envelope)
                if hasattr(store, "insert"):
                    store.insert(subject, env)
                elif hasattr(store, "insert_event"):
                    store.insert_event(env)
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "SSPBusEventPersistError",
                    {"subject": subject, "error": str(e)},
                )

    # -------------------------
    # Spans (start/end around a step)
    # -------------------------
    @asynccontextmanager
    async def span(
        self,
        *,
        context: Optional[Dict[str, Any]] = None,
        name: str,
        attrs: Optional[Dict[str, Any]] = None,
    ):
        """
        Usage:
          async with telemetry.span(context=ctx, name="actor.propose", attrs={"k":"v"}):
              ... work ...
        Emits:
          ssp.actor.propose.start
          ssp.actor.propose.end (with status + duration_ms)
        """
        if attrs is None:
            attrs = {}
        start_subject = f"{self.subject_root}.{name}.start"
        end_subject = f"{self.subject_root}.{name}.end"

        t0 = time.perf_counter()
        await self.publish(start_subject, {"attrs": attrs, "event": "start"}, context=context)
        status = "ok"
        err = None
        try:
            yield
        except Exception as e:
            status = "error"
            err = str(e)
            raise
        finally:
            dt_ms = int((time.perf_counter() - t0) * 1000)
            await self.publish(
                end_subject,
                {"attrs": attrs, "event": "end", "status": status, "duration_ms": dt_ms, "error": err},
                context=context,
            )
