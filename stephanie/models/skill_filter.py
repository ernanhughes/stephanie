# stephanie/models/skill_filter.py
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import \
    JSONB as JSON  # fallback to JSON if not PG
from sqlalchemy.orm import relationship

from stephanie.models.base import Base  # adjust import to your Base


class SkillFilterORM(Base):
    __tablename__ = "skill_filters"

    id = Column(String(64), primary_key=True)  # UUID/sha; you choose
    casebook_id = Column(String, ForeignKey("casebooks.id"), nullable=False)
    casebook = relationship("CaseBookORM", back_populates="skill_filters")

    domain = Column(String(32), index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    weight_delta_path = Column(String(256))
    weight_size_mb = Column(Float)

    vpm_residual_path = Column(String(256))
    vpm_preview_path = Column(String(256))

    alignment_score = Column(Float)
    improvement_score = Column(Float)
    stability_score = Column(Float)

    compatible_domains = Column(JSON)
    negative_interactions = Column(JSON)

    # Optional convenience fields (not in migration; add if wanted)
    vpm_min = Column(Float)
    vpm_max = Column(Float)
    vpm_metrics = Column(JSON)

    def __repr__(self):
        return f"<SkillFilter(id={self.id}, domain={self.domain})>"


    def get_compatible_domains(self) -> dict:
        """Safely get compatible domains as dict"""
        if not self.compatible_domains:
            return {}
        try:
            return json.loads(self.compatible_domains)
        except Exception:
            return {}
    
    def get_negative_interactions(self) -> list:
        """Safely get negative interactions as list"""
        if not self.negative_interactions:
            return []
        try:
            return json.loads(self.negative_interactions)
        except Exception: 
            return [] 