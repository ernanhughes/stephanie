# stephanie/models/pipeline_stage_orm.py
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class PipelineStageORM(Base):
    __tablename__ = 'pipeline_stages'

    # Primary Key
    id = Column(Integer, primary_key=True)

    # Goal and Run IDs
    goal_id = Column(String, nullable=True)
    run_id = Column(String, nullable=False, index=True)

    # Parent linkage for stage tree
    parent_stage_id = Column(Integer, ForeignKey('pipeline_stages.id'), nullable=True)

    # Reference to pipeline run and context state
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=True)
    input_context_id = Column(Integer, ForeignKey("context_states.id"), nullable=True)
    output_context_id = Column(Integer, ForeignKey("context_states.id"), nullable=True)

    # Stage metadata
    stage_name = Column(String, nullable=False)
    agent_class = Column(String, nullable=False)
    protocol_used = Column(String, nullable=False)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Status and scoring
    status = Column(String, nullable=False, index=True)  # accepted, rejected, retry, partial, pending
    score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)

    # Symbolic rules applied
    symbols_applied = Column(JSON, nullable=True)

    # Dynamic metadata (model name, config, etc.)
    extra_data = Column(JSONB, nullable=True)

    # Policy flags
    exportable = Column(String, nullable=True)
    reusable = Column(String, nullable=True)
    invalidated = Column(String, nullable=True)

    # Relationships
    children = relationship("PipelineStageORM", backref="parent", remote_side=[id])
    pipeline_run = relationship("PipelineRunORM", back_populates="stages")
    input_context = relationship("ContextStateORM", foreign_keys=[input_context_id])
    output_context = relationship("ContextStateORM", foreign_keys=[output_context_id])

    def __repr__(self):
        return f"<PipelineStageORM(id={self.id}, stage_name='{self.stage_name}', run_id='{self.run_id}', status='{self.status}')>"