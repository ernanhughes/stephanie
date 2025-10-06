# stephanie/types/training_event.py
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


class TrainingEventCreate(BaseModel):
    model_key: str
    dimension: str
    kind: str  # "pairwise" | "pointwise"
    query_text: str
    pos_text: Optional[str] = None
    neg_text: Optional[str] = None
    cand_text: Optional[str] = None
    label: Optional[int] = None
    weight: float = 1.0
    trust: float = 0.0
    goal_id: Optional[str] = None
    pipeline_run_id: Optional[int] = None
    agent_name: Optional[str] = None
    source: str = "memento"
    meta: Dict = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def _bin_label(cls, v):
        return None if v is None else int(1 if v else 0)

class TrainingEventUpdate(BaseModel):
    processed: Optional[bool] = None
    meta: Optional[Dict] = None

class TrainingEventOut(BaseModel):
    id: int
    model_key: str
    dimension: str
    kind: str
    query_text: str
    pos_text: Optional[str] = None
    neg_text: Optional[str] = None
    cand_text: Optional[str] = None
    label: Optional[int] = None
    weight: float
    trust: float
    source: str
    meta: Dict
