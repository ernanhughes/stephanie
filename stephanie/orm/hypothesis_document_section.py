# stephanie/orm/hypothesis_document_section.py
from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer

from stephanie.orm.base import Base


class HypothesisDocumentSectionORM(Base):
    __tablename__ = "hypothesis_document_section"

    id = Column(Integer, primary_key=True)
    hypothesis_id = Column(Integer, ForeignKey("hypothesis.id"), nullable=False)
    document_section_id = Column(
        Integer, ForeignKey("document_section.id"), nullable=False
    )
