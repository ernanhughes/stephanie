# models/document_section.py

from sqlalchemy import ARRAY, JSON, Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class DocumentSectionORM(Base):
    __tablename__ = "document_sections"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    section_name = Column(String, nullable=False)      # e.g., "Introduction", "Method"
    section_text = Column(Text, nullable=True)
    source = Column(Text, nullable=True)               # e.g., original text or system

    domains = Column(ARRAY(String))  
    embedding = Column(JSON, nullable=True)            # vector embedding if used
    summary = Column(Text, nullable=True)              # LLM-generated summary
    extra_data = Column(JSON, nullable=True)           # MR.Q scores, confidences, etc.

    document = relationship("DocumentORM", back_populates="sections")
    domains = relationship("DocumentSectionDomainORM", back_populates="document_section", cascade="all, delete-orphan")


    def to_dict(self):
        return {
            "id": self.id,
            "document_id": self.document_id,
            "section_name": self.section_name,
            "section_text": self.section_text,
            "summary": self.summary,
            "extra_data": self.extra_data,
        }

    def __repr__(self):
        return f"<Section[{self.document_id}] '{self.section_name}'>"
