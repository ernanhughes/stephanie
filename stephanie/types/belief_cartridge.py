# stephanie/types/belief_cartridge.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BeliefCartridge(BaseModel):
    """
    Pydantic type mirror of BeliefCartridgeORM.

    Used for serialization, API responses, and validation
    without coupling to SQLAlchemy internals.
    """

    model_config = ConfigDict(from_attributes=True)  # enables ORM -> model conversion

    id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    source_id: Optional[str] = None
    source_type: Optional[str] = None

    markdown_content: str
    goal_tags: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list)

    idea_payload: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = None
    is_active: bool = True

    derived_from: List[str] = Field(default_factory=list)
    applied_in: List[str] = Field(default_factory=list)
    version: int = 1

    memcube_id: Optional[str] = None
    debug_log: Optional[Dict[str, Any]] = None

    # ---- Optional attached evaluation summaries ----
    latest_scores: Optional[Dict[str, float]] = None
    epistemic_gain: Optional[float] = None

    # ---- Helpers (safe fallbacks) ----
    @classmethod
    def from_orm_with_scores(cls, orm_obj) -> "BeliefCartridge":
        """
        Converts ORM → Pydantic and includes computed fields if available.
        """
        model = cls.model_validate(orm_obj, from_attributes=True)

        # optional runtime enrichment
        if hasattr(orm_obj, "latest_score_dict"):
            model.latest_scores = orm_obj.latest_score_dict() or {}
        if hasattr(orm_obj, "epistemic_gain"):
            model.epistemic_gain = float(orm_obj.epistemic_gain()) or 0.0

        return model
