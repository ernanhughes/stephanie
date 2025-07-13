# stephanie/models/document_domain.py
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class DocumentDomainORM(Base):
    __tablename__ = "document_domains"

    id = Column(Integer, primary_key=True)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    domain = Column(String, nullable=False)
    score = Column(Float, nullable=False)

    # Optional: relationship to document
    document = relationship("DocumentORM", back_populates="domains_rel")

    def to_dict(self):
        return {
            "id": self.id,
            "document_id": self.document_id,
            "domain": self.domain,
            "score": self.score,
        }
