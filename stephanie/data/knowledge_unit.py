from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.utils.hash_utils import hash_text


@dataclass
class KnowledgeUnit:
    """
    Structured transient knowledge extracted from chat or document context.
    Built using AI models and linked to persistent memory via the knowledge graph.

    Designed to serve as input for retrieval, scoring, reasoning, and response generation.
    """
    text: str = ""
    domains: Dict[str, float] = field(default_factory=dict)
    phrases: List[str] = field(default_factory=list)
    anchors: List[Dict[str, Any]] = field(default_factory=list)
    entities: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    linked_kg_nodes: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "domains": self.domains,
            "phrases": self.phrases,
            "anchors": self.anchors,
            "entities": self.entities,
            "linked_kg_nodes": [
                {
                    "node_id": n.get("node_id"),
                    "text": n.get("text"),
                    "type": n.get("type"),
                    "score": float(n.get("score", 0))
                }
                for n in self.linked_kg_nodes
            ],
            "provenance": self.provenance,
            "stats": self.stats,
        }

    def to_scorable(self, target_type: str = "knowledge_unit") -> Scorable:
        """
        Convert this KnowledgeUnit into a Scorable object
        so it can enter the standard scoring pipeline.
        """
        scorable_id = self.provenance.get("scorable_id", hash_text(self.text)[:16])

        return Scorable(
            id=scorable_id,
            text=self.text,
            target_type=getattr(ScorableType, target_type.upper(), ScorableType.KNOWLEDGE_UNIT),
            meta={
                "source": self.provenance.get("source", "unknown"),
                "domains": list(self.domains.keys()),
                "entity_types": list(self.entities.keys()),
                "kg_link_count": len(self.linked_kg_nodes),
                "processing_duration_ms": self.stats.get("processing_duration_ms"),
                "timestamp": self.stats["timestamp"],
                "original": self.to_dict()  # full unit preserved
            }
        )


