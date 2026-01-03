# stephanie/orm/document_section_domain.py
from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from stephanie.orm.base import Base


class DocumentSectionDomainORM(Base):
    __tablename__ = "document_section_domains"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_section_id = Column(
        Integer, ForeignKey("document_sections.id", ondelete="CASCADE"), nullable=False
    )
    domain = Column(String, nullable=False)
    score = Column(Float, nullable=False)

    document_section = relationship("DocumentSectionORM", back_populates="domains")

    def to_dict(self):
        return {
            "id": self.id,
            "document_section_id": self.document_section_id,
            "domain": self.domain,
            "score": self.score,
        }
