# stephanie/models/casebook.py
from __future__ import annotations
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Text, JSON as SA_JSON
)
from sqlalchemy.orm import relationship
from stephanie.models.base import Base

# If you’re on Postgres and prefer JSONB:
# from sqlalchemy.dialects.postgresql import JSONB as SA_JSON


class CaseBookORM(Base):
    __tablename__ = "casebooks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, default="")

    # Optional scoping fields (if you use them)
    pipeline_run_id = Column(Integer, nullable=True, index=True)   # or pipeline_run_id: Integer
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
