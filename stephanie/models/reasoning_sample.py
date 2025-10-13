# stephanie/models/reasoning_sample.py
from __future__ import annotations
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from stephanie.models.base import Base

class ReasoningSampleORM(Base):
    """
    ORM wrapper for the reasoning_samples_view.
    ----------------------------------------------------
    This view unifies evaluations, goals, and scorable text
    from multiple types (document, section, plan_trace, etc.).
    It is read-only and used for model training, analysis,
    and reasoning data extraction (e.g., TinyRecursion, SICQL).
    """
    __tablename__ = "reasoning_samples_view"
    __table_args__ = {"extend_existing": True}

    # --- Evaluation metadata ---
    evaluation_id     = Column(Integer, primary_key=True)
    scorable_id       = Column(String)
    scorable_type     = Column(String)
    agent_name        = Column(String)
    model_name        = Column(String)
    evaluator_name    = Column(String)
    strategy          = Column(String)
    reasoning_strategy = Column(String)
    embedding_type    = Column(String)
    source            = Column(String)
    pipeline_run_id   = Column(String)
    symbolic_rule_id  = Column(String)
    extra_data        = Column(JSON)
    created_at        = Column(DateTime)

    # --- Goal and text content ---
    goal_text         = Column(Text)
    scorable_text     = Column(Text)
    document_title    = Column(Text)
    document_summary  = Column(Text)
    document_url      = Column(Text)
    section_name      = Column(Text)
    section_summary   = Column(Text)

    # --- Aggregates ---
    scores            = Column(JSON)
    attributes        = Column(JSON)

    # --- Utility method ---
    def to_dict(self) -> dict:
        """Convert ORM row → simple serializable dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "goal_text": self.goal_text,
            "scorable_text": self.scorable_text,
            "scores": self.scores or [],
            "attributes": self.attributes or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_training_sample(self) -> dict:
        """Format this record as a reasoning training example."""
        return {
            "x_text": (self.goal_text or "").strip(),
            "y_text": (self.sorable_text or "").strip(),
            "z_text": self._build_reflection_text(),
            "target_text": self._select_target_text(),
            "halt_target": 0,  # can be derived later
        }

    # (optional internal helper if you want reflection logic)
    def _build_reflection_text(self) -> str:
        lines = []
        if self.scores:
            for s in self.scores:
                if isinstance(s, dict) and s.get("rationale"):
                    lines.append(f"{s.get('dimension')}: {s['rationale']}")
        if self.attributes:
            for a in self.attributes:
                if isinstance(a, dict):
                    lines.append(f"{a.get('dimension')}: energy={a.get('energy')}, unc={a.get('uncertainty')}")
        return "\n".join(lines[:10])

    def _select_target_text(self) -> str:
        target = (self.document_summary or self.section_summary or self.sorable_text or "").strip()
        if self.scores:
            top = max(self.scores, key=lambda s: s.get("score", 0))
            if top.get("rationale"):
                target = top["rationale"]
        return target.strip()
