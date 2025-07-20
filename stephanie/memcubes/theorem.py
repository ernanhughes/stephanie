# stephanie/memcubes/theorem.py
from datetime import datetime
from typing import List

from stephanie.memcubes.memcube import MemCube
from stephanie.scoring.scorable import Scorable
from stephanie.utils.file_utils import hash_text


class Theorem:
    def __init__(
        self,
        id: str,
        premises: List[str],
        conclusion: str,
        score: float = 0.5,
        strength: float = 0.7,
        relevance: float = 0.8,
        source: str = "belief_graph",
        version: str = "v1",
        metadata: dict = None
    ):
        self.id = id
        self.premises = premises
        self.conclusion = conclusion
        self.score = score
        self.strength = strength
        self.relevance = relevance
        self.source = source
        self.version = version
        self.usage_count = 0
        self.created_at = datetime.utcnow()
        self.last_used = self.created_at
        self.metadata = metadata or {
            "energy_trace": [],
            "pipeline_origin": "auto"
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Theorem":
        """Reconstruct from serialized data"""
        return cls(
            id=data["id"],
            premises=data["premises"],
            conclusion=data["conclusion"],
            score=data.get("score", 0.5),
            strength=data.get("strength", 0.7),
            relevance=data.get("relevance", 0.8),
            source=data.get("source", "belief_graph"),
            version=data.get("version", "v1"),
            metadata=data.get("metadata", {})
        )
    
    def to_dict(self) -> dict:
        """Convert to serializable format"""
        return {
            "id": self.id,
            "premises": self.premises,
            "conclusion": self.conclusion,
            "score": self.score,
            "strength": self.strength,
            "relevance": self.relevance,
            "source": self.source,
            "version": self.version,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "metadata": self.metadata
        }
    
    def __str__(self):
        return f"Theorem({self.id}): {' → '.join(self.premises)} → {self.conclusion}"

