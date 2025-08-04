from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ScoreResult:
    """Core scoring result with flexible attributes dictionary"""

    # Core fields (required for all scoring operations)
    dimension: str
    score: float  # Primary score used for aggregation
    source: str  # Scorer name (e.g., "sicql", "svm", "contrastive_ranker")
    rationale: str = ""
    weight: float = 1.0

    # Flexible attributes dictionary (replaces separate ScoreAttributes)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization and storage"""
        result = {
            "dimension": self.dimension,
            "score": self.score,
            "source": self.source,
            "rationale": self.rationale,
            "weight": self.weight,
            "attributes": self.attributes,  # Include attributes directly
        }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreResult":
        """Reconstruct from dictionary"""
        return cls(
            dimension=data["dimension"],
            score=data["score"],
            source=data["source"],
            rationale=data.get("rationale", ""),
            weight=data.get("weight", 1.0),
            attributes=data.get("attributes", {}),
        )

    def __repr__(self):
        base = f"ScoreResult(dim='{self.dimension}', score={self.score:.4f}, source='{self.source}')"
        if self.attributes:
            metric_count = len(self.attributes)
            sample = ", ".join(list(self.attributes.keys())[:3])
            return f"{base} [attrs={metric_count} metrics: {sample}...]"
        return base
