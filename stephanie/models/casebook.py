# stephanie/models/casebook.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON as SA_JSON
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base

# If you’re on Postgres and prefer JSONB:
# from sqlalchemy.dialects.postgresql import JSONB as SA_JSON


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if isinstance(dt, datetime) else None


def _json_safe(val):
    """
    Make JSON-ish fields safe to serialize. Leaves dict/list/str/num/bool/None as-is,
    converts datetimes to ISO strings, and falls back to str() for unknowns.
    """
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, datetime):
        return _iso(val)
    if isinstance(val, dict):
        return {k: _json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(v) for v in val]
    # Numpy scalars etc.
    try:
        import numpy as np  # optional
        if isinstance(val, np.generic):
            return val.item()
    except Exception:
        pass
    return str(val)


class CaseBookORM(Base):
    __tablename__ = "casebooks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, default="")

    # Optional scoping fields (if you use them)
    pipeline_run_id = Column(Integer, nullable=True, index=True)
    agent_name   = Column(String, nullable=True, index=True)
    tag          = Column(String, nullable=False, default="default")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # --- relationships ---
    cases = relationship(
        "CaseORM",
        back_populates="casebook",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # -------- convenience --------
    def to_dict(
        self,
        *,
        include_cases: bool = False,
        case_summary: bool = True,
        include_counts: bool = True,
    ) -> dict:
        """
        Serialize the casebook.
        - include_cases: include related cases
        - case_summary: if True, each case is summarized (no scorables); if False, include full case dicts
        - include_counts: add counts (e.g., case_count) without loading cases if not already loaded
        """
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description or "",
            "pipeline_run_id": self.pipeline_run_id,
            "agent_name": self.agent_name,
            "tag": self.tag,
            "created_at": _iso(self.created_at),
        }

        if include_counts:
            # If collection not loaded, len(self.cases) may trigger a lazy load.
            # If you want to avoid that, gate it behind include_cases or detect .loaded
            try:
                data["case_count"] = len(self.cases)
            except Exception:
                data["case_count"] = None

        if include_cases:
            if case_summary:
                data["cases"] = [c.to_dict(include_scorables=False) for c in (self.cases or [])]
            else:
                data["cases"] = [c.to_dict(include_scorables=True) for c in (self.cases or [])]

        return data

    def __repr__(self) -> str:
        return f"<CaseBookORM id={self.id} name={self.name!r} tag={self.tag!r}>"


class CaseORM(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    casebook_id = Column(Integer, ForeignKey("casebooks.id", ondelete="CASCADE"), nullable=False)

    goal_id    = Column(String, nullable=False, index=True)
    goal_text  = Column(Text, nullable=False)
    agent_name = Column(String, nullable=False, index=True)

    mars_summary = Column(SA_JSON, nullable=False, default=dict)
    scores       = Column(SA_JSON, nullable=False, default=dict)
    meta         = Column(SA_JSON, nullable=False, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # --- relationships ---
    casebook = relationship("CaseBookORM", back_populates="cases")

    scorables = relationship(
        "CaseScorableORM",
        back_populates="case",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # -------- convenience --------
    def to_dict(
        self,
        *,
        include_scorables: bool = False,
    ) -> dict:
        """
        Serialize the case.
        - include_scorables: include related CaseScorable rows
        """
        data = {
            "id": self.id,
            "casebook_id": self.casebook_id,
            "goal_id": self.goal_id,
            "goal_text": self.goal_text,
            "agent_name": self.agent_name,
            "mars_summary": _json_safe(self.mars_summary),
            "scores": _json_safe(self.scores),
            "meta": _json_safe(self.meta),
            "created_at": _iso(self.created_at),
        }

        if include_scorables:
            data["scorables"] = [cs.to_dict() for cs in (self.scorables or [])]

        return data

    def __repr__(self) -> str:
        return f"<CaseORM id={self.id} goal_id={self.goal_id!r} agent={self.agent_name!r}>"


class CaseScorableORM(Base):
    __tablename__ = "case_scorables"
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    scorable_id = Column(String, nullable=False)   # stays NOT NULL
    scorable_type = Column(String, nullable=True)
    role = Column(String, nullable=False, default="input")
    rank = Column(Integer, nullable=True)          # ← matches new column
    meta = Column(SA_JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    case = relationship("CaseORM", back_populates="scorables")

    # -------- convenience --------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "role": self.role,
            "rank": self.rank,
            "meta": _json_safe(self.meta),
            "created_at": _iso(self.created_at),
        }

    def __repr__(self) -> str:
        return f"<CaseScorableORM id={self.id} case_id={self.case_id} scorable_id={self.scorable_id!r} role={self.role!r}>"
