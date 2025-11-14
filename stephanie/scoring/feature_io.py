from __future__ import annotations
from typing import Protocol, Dict, Any, List, Optional

from stephanie.scoring.scorable import Scorable  # unified model


class FeatureProvider(Protocol):
    """Hydrate partial features from any backend (DB, cache, vector store)."""
    async def hydrate(self, scorable: Scorable) -> Dict[str, Any]:
        """
        Returns any subset of:
          - domains: List[{"name": str, "score": float, "source": str}]
          - ner:     List[{"text": str, "type": str|None, "span": [int|None,int|None], "source": str}]
          - embeddings: {"global": List[float], ...}
          - metrics_vector: Dict[str, float]
          - near_identity: Dict[str, Any]
        """
        ...


class FeatureWriter(Protocol):
    """Persist newly computed features (only deltas) back to storage."""
    async def persist(self, scorable: Scorable, features: Dict[str, Any]) -> None:
        ...


class ScoringService(Protocol):
    """
    Score scorable against a goal. Returns ScoreBundle-like dict:
    {
      "results": {
         "<dimension>": {
             "score": float,
             "attributes": { "ppl": ..., "entropy": ..., ... },
             "vector": { "columns": [...], "values": [...] }
         },
         ...
      }
    }
    """
    async def score(
        self, 
        goal: Dict[str, Any],
        scorable: Scorable,
        dims: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        ...
