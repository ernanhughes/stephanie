# stephanie/models/evaluation_attribute.py
from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class EvaluationAttributeORM(Base):
    __tablename__ = "evaluation_attributes"

    id = Column(Integer, primary_key=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False)

    # Identity
    dimension = Column(String, nullable=False)  # e.g., "alignment"
    source = Column(String, nullable=False)      # e.g., "sicql", "mrq", "ebt"

    # Rich metrics per source
    raw_score = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)


    # Core SICQL metrics
    q_value = Column(Float, nullable=True)        # Q-head output
    v_value = Column(Float, nullable=True)        # V-head output
    advantage = Column(Float, nullable=True)      # Q - V
    pi_value = Column(Float, nullable=True)        # Policy head output (logits)

    # Advanced diagnostics
    entropy = Column(Float, nullable=True)        # Policy entropy
    uncertainty = Column(Float, nullable=True)    # |Q - V| gap
    td_error = Column(Float, nullable=True)      # Temporal difference error
    expected_return = Column(Float, nullable=True) # Discounted future reward

    # Raw policy outputs
    policy_logits = Column(JSON, nullable=True)   # Full policy distribution

    # Relationships
    evaluation = relationship("EvaluationORM", back_populates="attributes")

    extra = Column(JSON, nullable=True)  # For future extensibility
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "dimension": self.dimension,
            "source": self.source,
            "q": self.q_value,
            "v": self.v_value,
            "advantage": self.advantage,
            "pi": self.pi_value,
            "entropy": self.entropy,
            "uncertainty": self.uncertainty,
            "td_error": self.td_error,
            "expected_return": self.expected_return,
            "policy_logits": self.policy_logits
        }