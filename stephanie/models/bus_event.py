# stephanie/models/bus_event.py
from __future__ import annotations

from typing import Any, Dict

import sqlalchemy as sa
from sqlalchemy import (JSON, Column, Float, Integer, String)
from sqlalchemy.dialects.postgresql import UUID

from stephanie.models.base import Base


class BusEventORM(Base):
    __tablename__ = "bus_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    guid = sa.Column(UUID(as_uuid=True), unique=True,
                     server_default=sa.text("gen_random_uuid()"),
                     nullable=False)

    # Publisher metadata
    event_id = Column(
        String, nullable=True
    )  # publisher's unique id (may be null)
    subject = Column(
        String, nullable=False, index=True
    )  # e.g. stephanie.events.arena.run.round_end
    event = Column(String, nullable=True, index=True)  # e.g. "round_end"
    ts = Column(Float, nullable=False, index=True)  # epoch seconds

    # Linkages (nullable; individually indexed)
    run_id = Column(String, nullable=True, index=True)
    case_id = Column(String, nullable=True, index=True)
    paper_id = Column(String, nullable=True, index=True)
    section_name = Column(String, nullable=True, index=True)
    agent = Column(String, nullable=True, index=True)

    # Payloads
    # Use JSON so you can query bits if your backend supports it; on SQLite it maps to TEXT.
    payload_json = Column(
        JSON, nullable=False
    )  # original payload/body (as emitted)
    extras_json = Column(
        JSON, nullable=True
    )  # normalized/derived fields for fast charts

    # Dedupe / idempotency
    hash = Column(
        String, nullable=True, unique=True
    )  # sha256 of payload_json (or whole envelope)

    def to_dict(
        self, include_payload: bool = True, include_extras: bool = True
    ) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "event_id": self.event_id,
            "subject": self.subject,
            "event": self.event,
            "ts": self.ts,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "paper_id": self.paper_id,
            "section_name": self.section_name,
            "agent": self.agent,
            "hash": self.hash,
        }
        if include_payload:
            d["payload"] = self.payload_json
        if include_extras:
            d["extras"] = self.extras_json
        return d

    def __repr__(self) -> str:
        rid = f" run={self.run_id}" if self.run_id else ""
        return f"<BusEvent#{self.id} {self.subject} {self.event or ''}{rid} @ {self.ts}>"
