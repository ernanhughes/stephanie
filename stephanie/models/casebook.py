from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy import JSON as SA_JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from stephanie.models.base import Base
from stephanie.utils.date_utils import iso_date, utcnow
# ✅ Use the central sanitizer
from stephanie.utils.json_sanitize import sanitize  # <— NEW


class CaseBookORM(Base):
    __tablename__ = "casebooks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(Text, default="")
    pipeline_run_id = Column(Integer, nullable=True, index=True)
    agent_name = Column(String, nullable=True, index=True)
    tags = Column(JSONB, default=list)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    meta = Column(SA_JSON, nullable=True, default=dict)

    cases = relationship(
        "CaseORM",
        back_populates="casebook",
        cascade="all, delete-orphan",
        order_by="desc(CaseORM.created_at)",
        passive_deletes=True,
    )

    skill_filters = relationship("SkillFilterORM", back_populates="casebook")

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
            "tags": sanitize(self.tags) or [],          # <— sanitize
            "created_at": iso_date(self.created_at),
            "meta": sanitize(self.meta),                # <— sanitize
        }

        if include_counts:
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
        # fix: 'tag' doesn’t exist on this model
        return f"<CaseBookORM id={self.id} name={self.name!r} agent={self.agent_name!r}>"


class CaseORM(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    casebook_id = Column(Integer, ForeignKey("casebooks.id"), nullable=False)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=True)

    prompt_text = Column(Text, nullable=True)
    agent_name = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    meta = Column(SA_JSON, nullable=True)

    # Relationships
    scorables = relationship(
        "CaseScorableORM",
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="CaseScorableORM.rank.asc().nulls_last(), CaseScorableORM.id.asc()",
        passive_deletes=True,
    )
    casebook = relationship("CaseBookORM", back_populates="cases")

    def to_dict(self, *, include_scorables: bool = True) -> dict:
        data = {
            "id": self.id,
            "casebook_id": self.casebook_id,
            "goal_id": self.goal_id,
            "prompt_text": self.prompt_text or "",
            "agent_name": self.agent_name,
            "created_at": iso_date(self.created_at),
            "meta": sanitize(self.meta),               # <— sanitize
        }
        if include_scorables:
            data["scorables"] = [s.to_dict() for s in (self.scorables or [])]
        return data


class CaseScorableORM(Base):
    __tablename__ = "case_scorables"
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    scorable_id = Column(String, nullable=False)
    scorable_type = Column(String, nullable=True)
    role = Column(String, nullable=False, default="input")
    rank = Column(Integer, nullable=True)
    meta = Column(SA_JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    case = relationship("CaseORM", back_populates="scorables")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "role": self.role,
            "rank": self.rank,
            "meta": sanitize(self.meta),               # <— sanitize
            "created_at": iso_date(self.created_at),
        }


class CaseAttributeORM(Base):
    __tablename__ = "case_attributes"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    case_id     = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    key         = Column(Text, nullable=False, index=True)
    value_text  = Column(Text, nullable=True, index=True)
    value_num   = Column(Float, nullable=True, index=True)
    value_bool  = Column(Boolean, nullable=True, index=True)
    value_json  = Column(JSONB, nullable=True)
    created_at  = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    case = relationship("CaseORM", backref="attributes")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "key": self.key,
            "value_text": self.value_text,
            "value_num": self.value_num,
            "value_bool": self.value_bool,
            "value_json": sanitize(self.value_json),   # <— sanitize
            "created_at": iso_date(self.created_at),
        }
