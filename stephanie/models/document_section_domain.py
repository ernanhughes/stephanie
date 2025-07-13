# stephanie/models/document_section_domain.py
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class DocumentSectionDomainORM(Base):
    __tablename__ = "document_section_domains"

    id = Column(Integer, primary_key=True)
    document_section_id = Column(
        Integer, ForeignKey("document_sections.id", ondelete="CASCADE"), nullable=False
    )
    domain = Column(String, nullable=False)
    score = Column(Float, nullable=False)

    # Correct relationship name and back_populates link
    document_section = relationship("DocumentSectionORM", back_populates="domains")

    def to_dict(self):
        return {
            "id": self.id,
            "document_section_id": self.document_section_id,
            "domain": self.domain,
            "score": self.score,
        }
