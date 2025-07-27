# stephanie/models/goal.py
# models/goal.py
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class GoalORM(Base):
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True)
    goal_text = Column(String, nullable=False)
    goal_type = Column(String)
    focus_area = Column(String)
    strategy = Column(String)
    llm_suggested_strategy = Column(String)
    source = Column(String, default="user")
    difficulty = Column(String, default="medium")
    goal_category = Column(String, default="analyze")

    created_at = Column(DateTime, default=datetime.utcnow)

    prompts = relationship("PromptORM", back_populates="goal")
    hypotheses = relationship("HypothesisORM", back_populates="goal")
    pipeline_runs = relationship("PipelineRunORM", back_populates="goal")
    evaluations = relationship("EvaluationORM", back_populates="goal")
    lookaheads = relationship("LookaheadORM", back_populates="goal")
    reflection_deltas = relationship("ReflectionDeltaORM", back_populates="goal")
    ideas = relationship("IdeaORM", back_populates="goal")
    method_plans = relationship("MethodPlanORM", back_populates="goal")
    sharpening_predictions = relationship(
        "SharpeningPredictionORM", back_populates="goal"
    )
    symbolic_rules = relationship("SymbolicRuleORM", back_populates="goal")

    rule_applications = relationship(
        "RuleApplicationORM", back_populates="goal", cascade="all, delete-orphan"
    )
    search_hits = relationship(
        "SearchHitORM", back_populates="goal", cascade="all, delete-orphan"
    )
    cartridges = relationship(
        "CartridgeORM", back_populates="goal", cascade="all, delete-orphan"
    )
    training_stats = relationship(
        "TrainingStatsORM", back_populates="goal", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<GoalORM(id={self.id}, goal_text='{self.goal_text[:50]}')>"

    def to_dict(self):
        return {
            "id": self.id,
            "goal_text": self.goal_text,
            "goal_type": self.goal_type,
            "focus_area": self.focus_area,
            "strategy": self.strategy,
            "llm_suggested_strategy": self.llm_suggested_strategy,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    dimensions = relationship("GoalDimensionORM", back_populates="goal", cascade="all, delete-orphan")
