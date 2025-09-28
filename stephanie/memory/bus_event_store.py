# stephanie/memory/bus_event_store.py
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Union

from sqlalchemy import and_, func
from sqlalchemy.exc import IntegrityError

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.bus_event import BusEventORM


class BusEventStore(BaseSQLAlchemyStore):
    """
    Persistence helpers for bus_events.

    Use:
      - insert(subject: str, envelope: dict)      ← publish-style convenience
      - insert(row_dict: dict)                    ← raw row insert (already shaped for ORM)
      - upsert(row_dict: dict)                    ← idempotent write (by hash or (subject,event_id))

    Notes:
      - We persist the full *event body* into `data` (JSON).
      - Optionally persist the full *envelope* you published.
      - We extract linkage keys (run_id, case_id, paper_id, section_name, agent, event)
        to make UI queries simple.
    """

    orm_model = BusEventORM
    default_order_by = BusEventORM.ts.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "bus_events"

    def name(self) -> str:
        return self.name

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def insert(self, arg1: Union[str, Dict[str, Any]], envelope: Optional[Dict[str, Any]] = None) -> BusEventORM:
        """
        Overloaded:
          - insert(subject: str, envelope: dict)   ← from EventService.publish(...)
          - insert(row_dict: dict)                 ← raw ORM-ready dict
        """
        if isinstance(arg1, str) and isinstance(envelope, dict):
            row_dict = self._prepare_bus_row(arg1, envelope)
        elif isinstance(arg1, dict) and envelope is None:
            row_dict = arg1
        else:
            raise TypeError("insert() expects either (subject:str, envelope:dict) or (row_dict:dict).")

        def op(s):
            # de-dup by hash if present
            existing = None
            if row_dict.get("hash"):
                existing = s.query(BusEventORM).filter_by(hash=row_dict["hash"]).first()
            if existing is not None:
                return existing

            obj = BusEventORM(**row_dict)
            s.add(obj)
            s.flush()
            if self.logger:
                self.logger.log(
                    "BusEventInserted",
                    {"id": obj.id, "subject": obj.subject, "event": obj.event, "run_id": obj.run_id},
                )
            return obj

        return self._run(op)

    def upsert(self, row_dict: Dict[str, Any]) -> BusEventORM:
        """
        Idempotent write. Prefers 'hash'; falls back to (subject,event_id).
        """
        def op(s):
            q = None
            if row_dict.get("hash"):
                q = s.query(BusEventORM).filter_by(hash=row_dict["hash"])
            elif row_dict.get("subject") and row_dict.get("event_id"):
                q = s.query(BusEventORM).filter_by(
                    subject=row_dict["subject"], event_id=row_dict["event_id"]
                )

            existing = q.first() if q is not None else None
            if existing:
                for k, v in row_dict.items():
                    if k == "id":
                        continue
                    setattr(existing, k, v)
                if self.logger:
                    self.logger.log("BusEventUpdated", {"id": existing.id, "subject": existing.subject})
                return existing

            obj = BusEventORM(**row_dict)
            s.add(obj)
            s.flush()
            if self.logger:
                self.logger.log("BusEventInserted", {"id": obj.id, "subject": obj.subject})
            return obj

        return self._run(op)

    # -------------------------------------------------------------------------
    # Queries for UI
    # -------------------------------------------------------------------------
    def recent(
        self,
        limit: int = 200,
        run_id: Optional[str] = None,
        subject_like: Optional[str] = None,
        event: Optional[str] = None,
    ) -> List[BusEventORM]:
        """Most recent events, optionally filtered."""
        def op(s):
            q = s.query(BusEventORM)
            if run_id:
                q = q.filter(BusEventORM.run_id == str(run_id))
            if event:
                q = q.filter(BusEventORM.event == event)
            if subject_like:
                q = q.filter(BusEventORM.subject.like(subject_like))
            q = q.order_by(BusEventORM.ts.desc(), BusEventORM.id.desc()).limit(limit)
            return list(q.all())
        return self._run(op)

    def by_run(self, run_id: str, limit: int = 500) -> List[BusEventORM]:
        def op(s):
            return (
                s.query(BusEventORM)
                .filter(BusEventORM.run_id == str(run_id))
                .order_by(BusEventORM.ts.asc(), BusEventORM.id.asc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def by_case(self, case_id: str, limit: int = 500) -> List[BusEventORM]:
        def op(s):
            return (
                s.query(BusEventORM)
                .filter(BusEventORM.case_id == str(case_id))
                .order_by(BusEventORM.ts.asc(), BusEventORM.id.asc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def range_by_subjects(
        self,
        subjects: Iterable[str],
        ts_from: Optional[float] = None,
        ts_to: Optional[float] = None,
        limit: int = 1000,
    ) -> List[BusEventORM]:
        """Filter by a small set of subjects and optional time window."""
        subs = list(subjects)
        def op(s):
            q = s.query(BusEventORM).filter(BusEventORM.subject.in_(subs))
            if ts_from is not None:
                q = q.filter(BusEventORM.ts >= float(ts_from))
            if ts_to is not None:
                q = q.filter(BusEventORM.ts <= float(ts_to))
            q = q.order_by(BusEventORM.ts.asc(), BusEventORM.id.asc()).limit(limit)
            return q.all()
        return self._run(op)

    def delete_older_than(self, ts_cutoff: float) -> int:
        """Retention: delete rows older than `ts_cutoff` (epoch seconds)."""
        def op(s):
            q = s.query(BusEventORM).filter(BusEventORM.ts < float(ts_cutoff))
            n = q.delete(synchronize_session=False)
            if self.logger:
                self.logger.log("BusEventRetention", {"deleted": n, "cutoff": ts_cutoff})
            return n
        return self._run(op)

    def since_id(self, last_id: int, limit: int = 500) -> List[BusEventORM]:
        """Tail new rows after a known id (useful for polling UIs)."""
        def op(s):
            return (
                s.query(BusEventORM)
                .filter(BusEventORM.id > int(last_id))
                .order_by(BusEventORM.id.asc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def count_by_run(self, run_id: str) -> int:
        def op(s):
            return s.query(func.count(BusEventORM.id)).filter(BusEventORM.run_id == str(run_id)).scalar() or 0
        return self._run(op)

    def get(self, event_id: int) -> Optional[BusEventORM]:
        def op(s):
            return s.get(BusEventORM, event_id)
        return self._run(op)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _unwrap(envelope: Any) -> Dict[str, Any]:
        """
        Accepts the publish envelope. Returns the innermost event dict.
        Understands event-service style {payload:{ ... }}.
        """
        if not isinstance(envelope, dict):
            return {"raw": envelope}

        inner = envelope
        if "payload" in inner and isinstance(inner["payload"], dict):
            inner = inner["payload"]
        if "payload" in inner and isinstance(inner["payload"], dict) and (
            "event" in inner or "subject" in inner or "run_id" in inner
        ):
            inner = inner["payload"]

        return inner if isinstance(inner, dict) else {"raw": inner}

    @staticmethod
    def _to_str(x: Any) -> Optional[str]:
        if x is None:
            return None
        try:
            s = str(x)
            return s or None
        except Exception:
            return None

    def _prepare_bus_row(self, subject: str, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a publish envelope into a BusEventORM row dict.
        - Stores the innermost event body in `payload_json`
        - Optional derived/extra bits go into `extras_json`
        - Extracts linkage keys for UI (run_id/case_id/paper_id/section_name/agent/event)
        - Uses `ts` (float epoch seconds) per your ORM
        """
        now = float(envelope.get("timestamp") or envelope.get("ts") or time.time())
        body = self._unwrap(envelope)           # innermost dict (what you emitted)
        meta = body.get("meta") or {}

        run_id       = self._to_str(body.get("run_id") or body.get("arena_run_id") or meta.get("run_id"))
        case_id      = self._to_str(body.get("case_id") or meta.get("case_id"))
        paper_id     = self._to_str(body.get("paper_id") or meta.get("paper_id"))
        section_name = self._to_str(body.get("section_name") or meta.get("section_name"))
        agent        = self._to_str(body.get("agent") or meta.get("agent"))
        event        = self._to_str(body.get("event") or body.get("note") or "unknown")
        event_id     = self._to_str(envelope.get("event_id") or body.get("event_id"))

        # Stable-ish idempotency key (per-second bucket); include a hash of body for extra uniqueness
        hsrc = json.dumps(
            {
                "subject": subject,
                "event_id": event_id,
                "run_id": run_id,
                "event": event,
                "ts_bucket": int(now),
                "body_sha": hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(hsrc).hexdigest()

        # Optional small extras; keep it light so the row stays lean
        extras = {
            "service": envelope.get("service"),
            "instance": envelope.get("instance"),
            "schema": envelope.get("schema"),
            "publisher_ts": envelope.get("ts"),
        }

        return {
            "ts": now,
            "subject": subject,
            "event": event,
            "event_id": event_id,
            "run_id": run_id,
            "case_id": case_id,
            "paper_id": paper_id,
            "section_name": section_name,
            "agent": agent,
            "payload_json": body, 
            "extras_json": extras,
            "hash": digest,
        }
