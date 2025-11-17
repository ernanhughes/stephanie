from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

# Returned shapes
@dataclass
class NodeHit:
    node_id: str
    graph_id: str | None
    scorable_kind: str | None
    similarity: float
    meta: Dict[str, Any]

@dataclass
class EmbeddingBackend:
    get_embedding: Any
    query: Any

class SimilarityQueryEngine:
    def __init__(self, backend: EmbeddingBackend):
        self._b = backend

    def find_similar_nodes(
        self,
        text: str,
        top_k: int = 32,
        node_kinds: Optional[List[str]] = None,
        graph_ids: Optional[List[str]] = None,
        min_similarity: float = 0.0,
    ) -> List[NodeHit]:
        q = self._b.get_embedding(text, cfg="graph_nodes")
        results = self._b.query(
            q, top_k=top_k,
            filters={"node_kinds": node_kinds, "graph_ids": graph_ids},
        )
        hits: List[NodeHit] = []
        for r in results:
            sim = float(r.get("similarity", 0.0))
            if sim < min_similarity: 
                continue
            hits.append(NodeHit(
                node_id=r["node_id"],
                graph_id=r.get("graph_id"),
                scorable_kind=r.get("scorable_kind"),
                similarity=sim,
                meta=r.get("meta", {}),
            ))
        return hits

    # Convenience filters
    def find_similar_sections(self, text: str, **kw) -> List[NodeHit]:
        kw.setdefault("node_kinds", ["document_section"])
        return self.find_similar_nodes(text, **kw)

    def find_similar_turns(self, text: str, **kw) -> List[NodeHit]:
        kw.setdefault("node_kinds", ["conversation_turn"])
        return self.find_similar_nodes(text, **kw)
