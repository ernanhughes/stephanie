# stephanie/models/reflection.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, DateTime, Integer, String

from stephanie.models.base import Base


class ReflectionORM(Base):
    """
    Stores structured reflections produced by InformationReflectionAgent.

    We support multiple levels:
      - micro: single run / single draft
      - meso: topic-level aggregation
      - macro: cross-topic aggregation

    For the current loop you mainly use `level="micro"`.
    """

    __tablename__ = "reflections"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Logical identifiers
    task_id = Column(String(255), nullable=False, index=True)
    trace_id = Column(Integer, nullable=False, index=True)

    # "micro", "meso", "macro"
    level = Column(String(32), nullable=False, default="micro", index=True)

    # Content
    draft_text = Column(String, nullable=False)
    reference_text = Column(String, nullable=True)
    raw_text = Column(String, nullable=True)

    # Optional scalar score (e.g., quality score for this run)
    score = Column(Integer, nullable=True)

    # Structured reflection payloads
    problems = Column(JSON, nullable=False, default=list)      # list[dict]
    action_plan = Column(JSON, nullable=False, default=list)   # list[str]

    created_at = Column(DateTime, nullable=False, default=datetime.now)

    # ------------------------------------------------------------------ #
    # Convenience helpers (optional, but often handy)
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "level": self.level,
            "draft_text": self.draft_text,
            "reference_text": self.reference_text,
            "raw_text": self.raw_text,
            "score": self.score,
            "problems": self.problems or [],
            "action_plan": self.action_plan or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_micro(
        cls,
        *,
        task_id: str,
        trace_id: int,
        draft_text: str,
        reference_text: str | None,
        score: Optional[int],
        problems: List[Dict[str, Any]],
        action_plan: List[str],
        raw_text: str,
        level: str = "micro",
    ) -> ReflectionORM:
        return cls(
            task_id=task_id,
            trace_id=trace_id,
            level=level,
            draft_text=draft_text,
            reference_text=reference_text,
            score=score,
            problems=problems,
            action_plan=action_plan,
            raw_text=raw_text,
        )
