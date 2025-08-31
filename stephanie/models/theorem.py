# stephanie/models/theorem.py
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, Table, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base

theorem_cartridges = Table(
    "theorem_cartridges",
    Base.metadata,
    Column("theorem_id", Integer, ForeignKey("theorems.id"), primary_key=True),
    Column("cartridge_id", Integer, ForeignKey("cartridges.id"), primary_key=True),
)

class TheoremORM(Base):
    __tablename__ = "theorems"

    id = Column(Integer, primary_key=True)
    statement = Column(Text, nullable=False)
    proof = Column(Text, nullable=True)

    # Soft pointer to scorable_embeddings.id (no FK)
    embedding_id = Column(Integer, nullable=True)

    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True)
    pipeline_run = relationship("PipelineRunORM", back_populates="theorems")

    created_at = Column(DateTime, default=datetime.utcnow)

    cartridges = relationship(
        "CartridgeORM", secondary=theorem_cartridges, back_populates="theorems"
    )

    def to_dict(self, include_cartridges: bool = False) -> dict:
        data = {
            "id": self.id,
            "statement": self.statement,
            "proof": self.proof,
            "embedding_id": self.embedding_id,
            "pipeline_run_id": self.pipeline_run_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_cartridges:
            data["cartridges"] = [
                {"id": c.id, "title": c.title, "source_type": c.source_type, "source_uri": c.source_uri}
                for c in self.cartridges
            ]
        return data


class CartridgeORM(Base):
    __tablename__ = "cartridges"

    id = Column(Integer, primary_key=True)

    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=True)
    goal = relationship("GoalORM", back_populates="cartridges")

    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True)
    pipeline_run = relationship("PipelineRunORM", back_populates="cartridges")

    source_type = Column(Text, nullable=False)  # e.g., 'document', 'hypothesis'
    source_uri = Column(Text)
    markdown_content = Column(Text, nullable=False)

    # Soft pointer to scorable_embeddings.id (no FK / no relationship)
    embedding_id = Column(Integer, nullable=True)

    title = Column(Text)
    summary = Column(Text)
    sections = Column(JSON)
    triples = Column(JSON)
    domain_tags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    triples_rel = relationship(
        "CartridgeTripleORM", back_populates="cartridge", cascade="all, delete-orphan"
    )

    theorems = relationship(
        "TheoremORM", secondary="theorem_cartridges", back_populates="cartridges"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "pipeline_run_id": self.pipeline_run_id,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "markdown_content": self.markdown_content,
            "embedding_id": self.embedding_id,
            "title": self.title,
            "summary": self.summary,
            "sections": self.sections,
            "triples": self.triples,
            "domain_tags": self.domain_tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
