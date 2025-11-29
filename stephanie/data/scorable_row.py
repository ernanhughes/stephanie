# stephanie/data/scorable_row.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ScorableRow:
    """
    Canonical features row for a Scorable. This schema is the authoritative
    representation for:
      - ScorableProcessor
      - VPM / ZeroModel
      - Critic / Nexus
      - HRM / SICQL trainers
      - Metrics dashboards
      - Run manifests
    
    IMPORTANT:
    Any time new fields are added to RowBuilder, update this class.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    scorable_id: str
    scorable_type: Optional[str]

    conversation_id: Optional[str] = None
    external_id: Optional[str] = None
    order_index: Optional[int] = None

    # ------------------------------------------------------------------
    # Textual content
    # ------------------------------------------------------------------
    text: str = ""
    title: str = ""

    # ------------------------------------------------------------------
    # Base extracted features
    # ------------------------------------------------------------------
    near_identity: Dict[str, Any] = field(default_factory=dict)

    domains: List[Dict[str, Any]] = field(default_factory=list)
    ner: List[Dict[str, Any]] = field(default_factory=list)

    ai_score: Optional[float] = None
    star: Optional[Union[int, float]] = None
    goal_ref: Optional[str] = None

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    embed_global: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # Metrics (canonical vector)
    # ------------------------------------------------------------------
    metrics_columns: List[str] = field(default_factory=list)
    metrics_values: List[float] = field(default_factory=list)
    metrics_vector: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Additional metadata
    # ------------------------------------------------------------------
    agreement: Optional[float] = None
    stability: Optional[float] = None

    chat_id: Optional[str] = None
    turn_index: Optional[int] = None

    parent_scorable_id: Optional[str] = None
    parent_scorable_type: Optional[str] = None
    order_in_parent: Optional[int] = None

    # ------------------------------------------------------------------
    # Vision / ZeroModel / VPM signals
    # ------------------------------------------------------------------
    vision_signals: Optional[Any] = None        # CHW uint8 array or None
    vision_signals_meta: Dict[str, Any] = field(default_factory=dict)

    # VPM PNG (legacy)
    vpm_png: Optional[str] = None

    # ------------------------------------------------------------------
    # VisiCalc fields (optional)
    # ------------------------------------------------------------------
    visicalc_report: Dict[str, Any] = field(default_factory=dict)
    visicalc_features: Optional[List[float]] = None
    visicalc_feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Pipeline + provenance info
    # ------------------------------------------------------------------
    rollout: Dict[str, Any] = field(default_factory=dict)

    processor_version: str = "3.0"
    content_hash16: str = ""
    created_utc: float = 0.0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
