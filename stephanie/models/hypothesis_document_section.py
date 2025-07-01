from sqlalchemy import Column, ForeignKey, Integer

from stephanie.models.base import Base


class HypothesisDocumentSectionORM(Base):
    __tablename__ = "hypothesis_document_section"

    id = Column(Integer, primary_key=True)
    hypothesis_id = Column(Integer, ForeignKey("hypothesis.id"), nullable=False)
    document_section_id = Column(Integer, ForeignKey("document_section.id"), nullable=False)
