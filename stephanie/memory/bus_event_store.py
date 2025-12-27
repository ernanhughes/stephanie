# stephanie/memory/bus_event_store.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Union

from sqlalchemy import Integer, cast, func

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.bus_event import BusEventORM
from stephanie.models.goal import GoalORM
from stephanie.models.pipeline_run import PipelineRunORM
from stephanie.utils.hash_utils import hash_bytes, hash_text


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

    def _goal_from_payload(self, p: dict) -> dict:
        """
        Try to pull a human-friendly goal/title from a run's earliest payload.
        We look across common fields used by your arena pipeline.
        """
        if not isinstance(p, dict):
            return {"title": None, "goal": None, "paper_id": None, "section_name": None, "agent": None}

        meta = p.get("meta") or {}
        # title candidates
        title = (
            p.get("title")
            or p.get("paper_title")
            or meta.get("title")
            or meta.get("paper_title")
            or p.get("goal_title")
        )
        # goal/intent candidates
        goal = (
            p.get("goal")
            or p.get("arena_goal")
            or meta.get("goal")
            or (p.get("plan") or {}).get("goal")
        )

        return {
            "title": title,
            "goal": goal,
            "paper_id": str(p.get("paper_id") or meta.get("paper_id") or "") or None,
            "section_name": str(p.get("section_name") or meta.get("section_name") or "") or None,
            "agent": str(p.get("agent") or meta.get("agent") or "") or None,
        }

    def recent_runs(self, limit: int = 50) -> list[dict]:
        """
        Latest distinct run_ids with counts, first/last ts, last event,
        plus goal_text from the joined pipeline_runs/goals.
        """
        def op(s):
            # aggregate with joins
            agg = (
                s.query(
                    BusEventORM.run_id.label("run_id"),
                    func.count(BusEventORM.id).label("count"),
                    func.min(BusEventORM.ts).label("first_ts"),
                    func.max(BusEventORM.ts).label("last_ts"),
                    GoalORM.goal_text.label("goal_text"),
                ) 
                .join(PipelineRunORM, PipelineRunORM.id == cast(BusEventORM.run_id, Integer))
                .join(GoalORM, GoalORM.id == PipelineRunORM.goal_id)
                .filter(BusEventORM.run_id.isnot(None))
                .group_by(BusEventORM.run_id, GoalORM.goal_text)
                .order_by(func.max(BusEventORM.ts).desc())
                .limit(limit)
            ).all()

            run_ids = [r.run_id for r in agg]
            if not run_ids:
                return []

            # last event per run
            last_rows = (
                s.query(BusEventORM.run_id, BusEventORM.event)
                .filter(BusEventORM.run_id.in_(run_ids))
                .order_by(BusEventORM.run_id.asc(), BusEventORM.ts.desc(), BusEventORM.id.desc())
            ).all()
            last_event_by_run = {}
            for rid, ev in last_rows:
                if rid not in last_event_by_run:
                    last_event_by_run[rid] = ev

            # earliest payload per run
            earliest_rows = (
                s.query(BusEventORM)
                .filter(BusEventORM.run_id.in_(run_ids))
                .order_by(BusEventORM.run_id.asc(), BusEventORM.ts.asc(), BusEventORM.id.asc())
            ).all()
            first_payload_by_run = {}
            for row in earliest_rows:
                if row.run_id not in first_payload_by_run:
                    first_payload_by_run[row.run_id] = row.payload_json

            # build output
            out = []
            for r in agg:
                meta = self._goal_from_payload(first_payload_by_run.get(r.run_id, {}) or {})
                out.append({
                    "run_id": r.run_id,
                    "count": int(r.count or 0),
                    "first_ts": float(r.first_ts or 0),
                    "last_ts": float(r.last_ts or 0),
                    "last_event": last_event_by_run.get(r.run_id),
                    "title": meta.get("title"),
                    "goal": meta.get("goal"),
                    "goal_text": r.goal_text,   # <-- comes from the join
                    "paper_id": meta.get("paper_id"),
                    "section_name": meta.get("section_name"),
                    "agent": meta.get("agent"),
                })
            return out

        return self._run(op)

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

    def range_by_subjects(
        self,
        subjects: Iterable[str],
        ts_from: Optional[float] = None,
        ts_to: Optional[float] = None,
        limit: int = 1000,
    ) -> List[BusEventORM]:
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
        def op(s):
            q = s.query(BusEventORM).filter(BusEventORM.ts < float(ts_cutoff))
            n = q.delete(synchronize_session=False)
            if self.logger:
                self.logger.log("BusEventRetention", {"deleted": n, "cutoff": ts_cutoff})
            return n
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

    def payloads_by_run(self, run_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        """
        Return enriched + flattened event payloads for a run.
        - Preserves ORM metadata (event, subject, run_id, etc.)
        - Flattens payload fields into the dict, but never overwrites core fields
        - Keeps full payload under `payload`
        """
        def op(s):
            rows = (
                s.query(BusEventORM)
                .filter(BusEventORM.run_id == str(run_id))
                .order_by(BusEventORM.ts.asc(), BusEventORM.id.asc())
                .limit(limit)
                .all()
            )

            out = []
            for row in rows:
                enriched = {
                    "id": row.id,
                    "guid": str(row.guid),
                    "ts": row.ts,
                    "event": row.event,
                    "subject": row.subject,
                    "event_id": row.event_id,
                    "run_id": row.run_id,
                    "case_id": row.case_id,
                    "paper_id": row.paper_id,
                    "section_name": row.section_name,
                    "agent": row.agent,
                    "extras": row.extras_json or {},
                }

                payload = row.payload_json or {}
                if isinstance(payload, dict):
                    # Fallback injection for initial_scored without topk
                    if payload.get("event") == "initial_scored" and "topk" not in payload:
                        payload["topk"] = [{
                            "case_id": row.case_id,
                            "guid": str(row.guid),
                            "origin": payload.get("origin", "unknown"),
                            "variant": payload.get("variant", "—"),
                            "overall": payload.get("best_overall"),
                            "k": payload.get("marginal_per_ktok"),
                            "verified": False,
                        }]

                    for k, v in payload.items():
                        if k in (
                            "id", "ts", "event", "subject", "event_id",
                            "run_id", "case_id", "paper_id", "section_name", "agent"
                        ):
                            continue
                        enriched[k] = v

                    if "topk" in payload and isinstance(payload["topk"], list):
                        for t in payload["topk"]:
                            if isinstance(t, dict):
                                if "case_id" not in t or not t["case_id"]:
                                    t["case_id"] = row.case_id
                                if "guid" not in t or not t["guid"]:
                                    t["guid"] = str(row.guid)

                enriched["payload"] = payload
                out.append(enriched)
            return out

        return self._run(op)

    def last_event_for_run(self, run_id: str) -> Optional[str]:
        def op(s):
            row = (
                s.query(BusEventORM.event)
                .filter(BusEventORM.run_id == str(run_id))
                .order_by(BusEventORM.ts.desc(), BusEventORM.id.desc())
                .limit(1)
                .first()
            )
            return row[0] if row else None
        return self._run(op)

    def runs_summary(self, limit: int = 50) -> list[dict]:
        """Latest N runs by last event time."""
        def op(s):
            rows = (
                s.query(
                    BusEventORM.run_id.label("run_id"),
                    func.count(BusEventORM.id).label("count"),
                    func.min(BusEventORM.ts).label("first_ts"),
                    func.max(BusEventORM.ts).label("last_ts"),
                    func.max(BusEventORM.event).label("last_event"),
                )
                .filter(BusEventORM.run_id.isnot(None))
                .group_by(BusEventORM.run_id)
                .order_by(func.max(BusEventORM.ts).desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "run_id": r.run_id,
                    "count": int(r.count or 0),
                    "first_ts": float(r.first_ts or 0.0),
                    "last_ts": float(r.last_ts or 0.0),
                    "last_event": r.last_event,
                }
                for r in rows
            ]
        return self._run(op)

    # ------------------------------------------------------
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
                "body_sha": hash_text(json.dumps(body, sort_keys=True, ensure_ascii=False)),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        digest = hash_bytes(hsrc)

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
