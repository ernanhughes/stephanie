from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, Table, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base

# Association table for many-to-many relationship
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
    embedding_id = Column(Integer, ForeignKey("embeddings.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship with Cartridges
    cartridges = relationship(
        "CartridgeORM",
        secondary=theorem_cartridges,
        back_populates="theorems"
    )

    def to_dict(self, include_cartridges: bool = False) -> dict:
        data = {
            "id": self.id,
            "statement": self.statement,
            "proof": self.proof,
            "embedding_id": self.embedding_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

        if include_cartridges:
            data["cartridges"] = [
                {
                    "id": c.id,
                    "title": c.title,
                    "source_type": c.source_type,
                    "source_uri": c.source_uri
                }
                for c in self.cartridges
            ]

        return data

class CartridgeORM(Base):
    __tablename__ = 'cartridges'

    id = Column(Integer, primary_key=True)

    # Association
    goal_id = Column(Integer, ForeignKey('goals.id'), nullable=True)  # Optional link to goal
    goal = relationship("GoalORM", back_populates="cartridges")

    # Source metadata
    source_type = Column(Text, nullable=False)     # e.g., 'document', 'hypothesis', 'response'
    source_uri = Column(Text)                      # Original file / API reference
    markdown_content = Column(Text, nullable=False)   # Where the rendered content lives
    embedding_id = Column(Integer)  # <-- New embedding reference
   

    # Core content
    title = Column(Text)
    summary = Column(Text)
    sections = Column(JSON)                          # {"Intro": "...", "Conclusion": "..."}
    triples = Column(JSON)                           # e.g., [("LLMs", "can be fine-tuned with", "LoRA")]

    # Domains
    domain_tags = Column(JSON)                       # e.g., ["machine learning", "ethics"]

    created_at = Column(DateTime, default=datetime.utcnow)

    domains_rel = relationship("CartridgeDomainORM", back_populates="cartridge", cascade="all, delete-orphan")

    triples_rel = relationship("CartridgeTripleORM", back_populates="cartridge", cascade="all, delete-orphan")

    theorems = relationship(
        "TheoremORM",
        secondary="theorem_cartridges",
        back_populates="cartridges")
    

    def to_dict(self):
        return {
            "id": self.id,
            "goal_id": self.goal_id,
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
            "domains": [
                {
                    "domain": rel.domain,
                    "score": rel.score
                }
                for rel in self.domains_rel
            ]
        }

