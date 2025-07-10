from dataclasses import dataclass
from typing import Optional

from stephanie.scoring.scorable_factory import TargetType


@dataclass
class ScoreResult:
    """
    Represents the result of scoring a single dimension, including the score,
    rationale text, and weight used in aggregation.
    """
    dimension: str
    score: float
    weight: float
    rationale: str
    source: str
    target_type: str = "custom"
    prompt_hash: Optional[str] = None
    parser_error: Optional[str] = None

    def weighted(self) -> float:
        return self.score * self.weight

    def to_dict(self):
        return {
            "dimension": self.dimension,
            "score": self.score,
            "weight": self.weight,
            "rationale": self.rationale,
            "prompt_hash": self.prompt_hash,
            "source": self.source,
            "parser_error": self.parser_error,
            "target_type": self.target_type.value if isinstance(self.target_type, TargetType) else self.target_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreResult":
        return cls(
            dimension=data.get("dimension"),
            score=data.get("score"),
            weight=data.get("weight", 1.0),
            rationale=data.get("rationale", ""),
            source=data.get("source", ""),
            prompt_hash=data.get("prompt_hash", ""),
            parser_error=data.get("parser_error", None),
            target_type=data.get("target_type", "custom"),    
        )
