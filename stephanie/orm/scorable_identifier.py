# stephanie/orm/identifier.py
from __future__ import annotations

from sqlalchemy import (Column, DateTime, Index, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.sql import func

from stephanie.orm.base import Base  # adjust to your declarative Base import


class IdentifierORM(Base):
    __tablename__ = "identifiers"

    id = Column(Integer, primary_key=True)
    identifier_type = Column(String(64), nullable=False, index=True)
    identifier_value = Column(Text, nullable=False)  # <-- Text avoids truncation pain
    name = Column(String(256), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("identifier_type", "identifier_value", name="uq_identifier_type_value"),
        # optional: faster lookups
        Index("ix_identifiers_type_value", "identifier_type", "identifier_value"),
    )


class ScorableIdentifierORM(Base):
    """
    Canonical identifier record, e.g.
      - type='arxiv', value='2506.21734'
      - type='url',   value='https://arxiv.org/abs/2506.21734'
      - type='doi',   value='10.1145/...'
      - type='doc_id',value='internal-123'
    """
    __tablename__ = "identifiers"

    id = Column(Integer, primary_key=True)

    # what kind of identifier is this?
    identifier_type = Column(Text, nullable=False, index=True)

    # the raw identifier string (this is the “name/value” you mentioned)
    identifier_value = Column(Text, nullable=False, index=True)

    # optional: human-friendly label and description
    name = Column(Text, nullable=True)
    description = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("identifier_type", "identifier_value", name="uq_identifier_type_value"),
    )
