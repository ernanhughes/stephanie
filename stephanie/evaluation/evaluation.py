# stephanie/evaluation/evaluation.py
"""Canonical Evaluation (§8) + generic evaluation attributes (§17)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, MutableMapping, Optional

from stephanie.evaluation.confidence import validate_confidence
from stephanie.evaluation.criterion import Criterion
from stephanie.evaluation.interpretation import Interpretation
from stephanie.evaluation.subject import SubjectRef


@dataclass(frozen=True)
class EvaluatorRef:
    name: str
    version: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    evaluation_id: str

    subject: SubjectRef
    criterion: Criterion

    evaluator: EvaluatorRef

    created_at: datetime

    confidence: Optional[float] = None
    confidence_source: Optional[str] = None

    interpretation: Optional[Interpretation] = None

    run_id: Optional[str] = None
    experiment_id: Optional[str] = None

    model_id: Optional[str] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    is_active: bool = True

    supersedes_id: Optional[str] = None

    def __post_init__(self) -> None:
        validate_confidence(self.confidence)
        if self.run_id is not None:
            object.__setattr__(self, "run_id", str(self.run_id))
        if self.experiment_id is not None:
            object.__setattr__(self, "experiment_id", str(self.experiment_id))
        if self.model_id is not None:
            object.__setattr__(self, "model_id", str(self.model_id))

    def superseded_by(self, new_id: str) -> "Evaluation":
        """Build the successor record (append/supersede lifecycle, §20)."""
        from dataclasses import replace

        return replace(
            self,
            evaluation_id=new_id,
            supersedes_id=self.evaluation_id,
            created_at=datetime.utcnow(),
        )


@dataclass
class EvaluationObservation:
    """Pre-persistence input to EvaluationRuntime.record (§22)."""

    subject: SubjectRef
    criterion: Criterion
    evaluator: EvaluatorRef

    scores: list = field(default_factory=list)  # list[Score]; see score.py

    confidence: Optional[float] = None
    confidence_source: Optional[str] = None

    interpretation: Optional[Interpretation] = None

    run_id: Optional[str] = None
    experiment_id: Optional[str] = None

    model_id: Optional[str] = None

    evidence: list = field(default_factory=list)  # list[EvidenceRef]

    attributes: list = field(default_factory=list)  # list[EvaluationAttribute]

    score_attributes: list = field(default_factory=list)  # list[ScoreAttribute]

    provenance: Any = None  # EvaluationProvenance; see provenance.py

    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationAttribute:
    """Generic EAV attribute (§17): namespace.name, e.g. sicql.q_value."""

    evaluation_id: str

    namespace: str
    name: str

    value: Any

    source: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}.{self.name}"
