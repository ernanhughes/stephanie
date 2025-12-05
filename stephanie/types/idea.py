from pydantic import BaseModel, Field
from typing import List, Dict, Any
from stephanie.utils.id_utils import generate_uuid  # or wherever your UUID util lives

class Idea(BaseModel):
    id: str = Field(default_factory=generate_uuid)
    title: str
    hypothesis: str
    method: str
    impact_summary: str

    concept_ids: List[str] = Field(default_factory=list)
    gap_ids: List[str] = Field(default_factory=list)

    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    utility_score: float = 0.0
    risk_score: float = 0.0
    r_final: float = 0.0

    generator_model: str
    prompt_hash: str

    critique_trace: Dict[str, Any] = Field(default_factory=dict)
