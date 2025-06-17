# models/document_section.py

from sqlalchemy import Column, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship

from co_ai.models.base import Base


class DocumentSectionORM(Base):
    __tablename__ = "document_sections"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    section_name = Column(String, nullable=False)
    section_text = Column(Text)
    source = Column(Text)

    domain = Column(String)                # e.g., "co_ai.reasoning"
    embedding = Column(JSON)               # optional vector embedding
    summary = Column(Text)                 # optional LLM-generated summary
    extra_data = Column(JSON)                # score details or LLM confidence

    document = relationship("DocumentORM", back_populates="sections")

    def to_dict(self):
        return {
            "id": self.id,
            "document_id": self.document_id,
            "section_name": self.section_name,
            "section_text": self.section_text,
        }

    def __repr__(self):
        return f"<Section[{self.document_id}] '{self.section_name}'>"
