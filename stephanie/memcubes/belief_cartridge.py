# stephanie/models/belief_cartridge.py
from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class BeliefCartridgeORM(Base):
    __tablename__ = "belief_cartridges"
    
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Source information
    source_id = Column(String, index=True)
    source_type = Column(String, index=True)  # e.g., "paper", "blog", "experiment"
    source_url = Column(String)
    
    # Core content
    markdown_content = Column(Text, nullable=False)
    goal_tags = Column(JSON, default=list, nullable=False)  # Stored as JSON array
    domain_tags = Column(JSON, default=list, nullable=False)  # Stored as JSON array
    idea_payload = Column(JSON)  # Can be JSON blob or link to structured table
    rationale = Column(Text)
    is_active = Column(Boolean, default=True, index=True)
    
    # Provenance and application history
    derived_from = Column(JSON, default=list)  # List of belief IDs
    applied_in = Column(JSON, default=list)  # List of world IDs or run IDs
    version = Column(Integer, default=1)
    
    # Optional attachments
    memcube_id = Column(String, index=True)  # Reference to MemCube
    debug_log = Column(JSON)  # Dictionary of debug info
    
    # Relationships
    goal_id = Column(Integer, ForeignKey("goals.id", ondelete="SET NULL"), nullable=True)
    goal = relationship("GoalORM", back_populates="belief_cartridges")
    
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    document = relationship("DocumentORM", back_populates="belief_cartridges")
    
    # Track usage
    used_in_pipelines = relationship("PipelineRunORM", back_populates="belief_cartridges")
    evaluations = relationship("EvaluationORM", back_populates="belief_cartridge")

    def __repr__(self):
        return f"<BeliefCartridgeORM(id={self.id}, source_id={self.source_id}, created_at={self.created_at})>"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "markdown_content": self.markdown_content,
            "goal_tags": self.goal_tags,
            "domain_tags": self.domain_tags,
            "idea_payload": self.idea_payload,
            "rationale": self.rationale,
            "is_active": self.is_active, 
            "derived_from": self.derived_from,
            "applied_in": self.applied_in,
            "version": self.version,
            "memcube_id": self.memcube_id,
            "debug_log": self.debug_log
        }
    