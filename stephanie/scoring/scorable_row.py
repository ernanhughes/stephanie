# stephanie/scoring/scorable_row.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ScorableRow:
    """
    Canonical features row for a Scorable, with a stable, explicit schema.
    Used by:
      - ScorableProcessor
      - NexusGraph
      - VPM / ZeroModel
      - Trainers / evaluators

    Extend this class when you add new columns; everywhere else just
    consumes it.
    """
    scorable_id: str
    scorable_type: Optional[str]

    conversation_id: Optional[str] = None
    external_id: Optional[str] = None
    order_index: Optional[int] = None

    text: str = ""
    title: str = ""

    near_identity: Dict[str, Any] = field(default_factory=dict)
    domains: List[Dict[str, Any]] = field(default_factory=list)
    ner: List[Dict[str, Any]] = field(default_factory=list)

    ai_score: Optional[float] = None
    star: Optional[Union[int, float]] = None
    goal_ref: Optional[str] = None

    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    embed_global: Optional[List[float]] = None

    metrics_columns: List[str] = field(default_factory=list)
    metrics_values: List[float] = field(default_factory=list)
    metrics_vector: Dict[str, float] = field(default_factory=dict)

    agreement: Optional[float] = None
    stability: Optional[float] = None
    chat_id: Optional[str] = None
    turn_index: Optional[int] = None

    parent_scorable_id: Optional[str] = None
    parent_scorable_type: Optional[str] = None
    order_in_parent: Optional[int] = None

    vpm_png: Optional[str] = None
    rollout: Dict[str, Any] = field(default_factory=dict)

    processor_version: str = "2.0"
    content_hash16: str = ""
    created_utc: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
