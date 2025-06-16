# models/method_plan.py
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String)
from sqlalchemy.orm import relationship

from .base import Base


class MethodPlanORM(Base):
    __tablename__ = "method_plans"

    id = Column(Integer, primary_key=True)

    # Idea source
    idea_text = Column(String, nullable=False)
    idea_id = Column(Integer, ForeignKey("ideas.id"), nullable=True)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False)

    # Research design fields
    research_objective = Column(String, nullable=False, default="research")
    key_components = Column(JSON)  # List of technical components or modules
    experimental_plan = Column(String)
    hypothesis_mapping = Column(JSON)  # e.g., {"H1": "handled by introspection module"}
    search_strategy = Column(String)
    knowledge_gaps = Column(String)
    next_steps = Column(String)

    # Supporting metadata
    task_description = Column(String)
    baseline_method = Column(String)
    literature_summary = Column(String)
    code_plan = Column(String)  # Optional pseudocode or starter code
    focus_area = Column(String)  # e.g., chemistry, nlp, cv, meta_learning
    strategy = Column(String)  # e.g., graph_attention_with_positional_embeddings

    # Scoring system
    score_novelty = Column(Float)
    score_feasibility = Column(Float)
    score_impact = Column(Float)
    score_alignment = Column(Float)

    # Evolution tracking
    evolution_level = Column(Integer, default=0)
    parent_plan_id = Column(Integer, ForeignKey("method_plans.id"), nullable=True)
    is_refinement = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    # Relationships
    goal = relationship("GoalORM", back_populates="method_plans")
    parent_plan = relationship("MethodPlanORM", remote_side=[id], backref="refinements")

    def to_dict(self, include_relationships: bool = False) -> dict:
        result = {
            "id": self.id,
            "idea_text": self.idea_text,
            "idea_id": self.idea_id,
            "goal_id": self.goal_id,
            "research_objective": self.research_objective,
            "key_components": self.key_components,
            "experimental_plan": self.experimental_plan,
            "hypothesis_mapping": self.hypothesis_mapping,
            "search_strategy": self.search_strategy,
            "knowledge_gaps": self.knowledge_gaps,
            "next_steps": self.next_steps,
            "baseline_method": self.baseline_method,
            "literature_summary": self.literature_summary,
            "code_plan": self.code_plan,
            "focus_area": self.focus_area,
            "strategy": self.strategy,
            "score_novelty": self.score_novelty,
            "score_feasibility": self.score_feasibility,
            "score_impact": self.score_impact,
            "score_alignment": self.score_alignment,
            "evolution_level": self.evolution_level,
            "parent_plan_id": self.parent_plan_id,
            "is_refinement": self.is_refinement,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

        if include_relationships and self.goal:
            result["goal"] = self.goal.to_dict()
            result["refinements"] = [r.to_dict() for r in self.refinements]

        return result