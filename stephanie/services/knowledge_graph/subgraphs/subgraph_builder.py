# stephanie/services/knowledge_graph/subgraphs/subgraph_builder.py
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from .edge_index import JSONLEdgeIndex, _rel_get
from .seed_finder import EmbeddingSeedFinder


@dataclass(frozen=True)
class SubgraphConfig:
    seed_k: int = 30
    per_entity_k: int = 12
    k_entities: int = 12

    max_hops: int = 2
    max_nodes: int = 200
    max_edges: int = 800

    # important safety knobs
    min_confidence: float = 0.75
    require_evidence: bool = True
    allowed_edge_types: Optional[List[str]] = None
    include_reverse: bool = True

    # seed control
    min_seed_score: float = 0.0

    # stop “hub explosion”
    max_incident_edges_per_node: int = 200


class SubgraphBuilder:
    """
    Responsible only for:
      - running bounded BFS expansion from seeds
      - applying filters (confidence/evidence/type)
      - producing stable output + stats

    It does NOT:
      - load edges (edge_index does)
      - decide seeds (seed_finder does)
    """

    def __init__(
        self,
        *,
        seed_finder: EmbeddingSeedFinder,
        edge_index: JSONLEdgeIndex,
        nexus_store: Any = None,
        logger: Any = None,
    ) -> None:
        self.seed_finder = seed_finder
        self.edge_index = edge_index
        self.nexus_store = nexus_store
        self.logger = logger

    def _node_label(self, node_id: str) -> str:
        if not self.nexus_store:
            return node_id
        try:
            n = self.nexus_store.get_node(node_id)
            if not n:
                return node_id
            for attr in ("name", "title", "text"):
                v = getattr(n, attr, None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            payload = getattr(n, "payload", None)
            if isinstance(payload, dict):
                nm = payload.get("name") or payload.get("title") or payload.get("text")
                if isinstance(nm, str) and nm.strip():
                    return nm.strip()
        except Exception as ex:
            if self.logger:
                self.logger.debug(f"SubgraphBuilder: label lookup failed for {node_id}: {ex}")
        return node_id

    def _has_evidence(self, e: Dict[str, Any]) -> bool:
        return bool(
            e.get("doc_hash")
            or e.get("evidence_type")
            or e.get("sentence_ix") is not None
            or e.get("scorable_id")
        )

    def _edge_key(self, s: str, t: str, r: str, e: Dict[str, Any]) -> Tuple[str, str, str, str]:
        """
        Key that dedupes edges but still distinguishes different evidence sources.
        If you later want “collapse evidence”, do it in post-processing.
        """
        doc_hash = str(e.get("doc_hash") or "")
        sent_ix = str(e.get("sentence_ix") if e.get("sentence_ix") is not None else "")
        return (s, t, r, f"{doc_hash}:{sent_ix}")

    def build(self, *, query: str, cfg: SubgraphConfig) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"nodes": [], "edges": [], "meta": {"reason": "empty_query"}}

        seed_info = self.seed_finder.find_seeds(
            query=query,
            seed_k=cfg.seed_k,
            per_entity_k=cfg.per_entity_k,
            k_entities=cfg.k_entities,
            min_seed_score=cfg.min_seed_score,
            max_seeds=cfg.max_nodes,  # never seed > max_nodes
        )
        seeds: List[str] = seed_info["seeds"]
        seed_terms: List[str] = seed_info["seed_terms"]

        if not seeds:
            return {
                "nodes": [],
                "edges": [],
                "meta": {
                    "query": query,
                    "seed_terms": seed_terms,
                    "seed_count": 0,
                    "reason": "no_seeds",
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            }

        allowed = set(cfg.allowed_edge_types or [])

        kept_nodes: Set[str] = set(seeds)
        kept_edges: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

        frontier: List[str] = list(seeds)

        for _hop in range(int(cfg.max_hops)):
            if not frontier:
                break

            next_frontier: List[str] = []

            for nid in frontier:
                # Bound the explosion from hubs
                incident = list(self.edge_index.neighbors(nid, include_reverse=cfg.include_reverse))
                if len(incident) > int(cfg.max_incident_edges_per_node):
                    incident = incident[: int(cfg.max_incident_edges_per_node)]

                for e in incident:
                    s = str(_rel_get(e, "source", "source_id", default=""))
                    t = str(_rel_get(e, "target", "target_id", default=""))
                    r = str(_rel_get(e, "type", "rel_type", default=""))

                    if not s or not t or not r:
                        continue

                    if allowed and r not in allowed:
                        continue

                    conf = float(_rel_get(e, "confidence", default=0.0))
                    if conf < float(cfg.min_confidence):
                        continue

                    if cfg.require_evidence and not self._has_evidence(e):
                        continue

                    key = self._edge_key(s, t, r, e)
                    if key not in kept_edges:
                        kept_edges[key] = {
                            "source": s,
                            "target": t,
                            "type": r,
                            "confidence": conf,
                            "doc_hash": e.get("doc_hash"),
                            "sentence_ix": e.get("sentence_ix"),
                            "scorable_id": e.get("scorable_id"),
                            "scorable_type": e.get("scorable_type"),
                            "evidence_type": e.get("evidence_type"),
                            "ts": e.get("ts"),
                        }
                        if len(kept_edges) >= int(cfg.max_edges):
                            break

                    # add nodes + expand
                    for other in (s, t):
                        if other and other not in kept_nodes and len(kept_nodes) < int(cfg.max_nodes):
                            kept_nodes.add(other)
                            next_frontier.append(other)

                if len(kept_edges) >= int(cfg.max_edges):
                    break

            frontier = next_frontier
            if len(kept_nodes) >= int(cfg.max_nodes) or len(kept_edges) >= int(cfg.max_edges):
                break

        # Stable output
        nodes = [{"id": nid, "label": self._node_label(nid)} for nid in sorted(kept_nodes)]
        edges = sorted(
            kept_edges.values(),
            key=lambda x: (x["source"], x["target"], x["type"], str(x.get("doc_hash") or ""), str(x.get("sentence_ix") or "")),
        )

        # Stats
        etypes = Counter(e["type"] for e in edges)
        evidence_count = sum(1 for e in edges if self._has_evidence(e))
        confidences = sorted(float(e.get("confidence", 0.0)) for e in edges)

        def q(p: float) -> float:
            if not confidences:
                return 0.0
            i = int((len(confidences) - 1) * p)
            return float(confidences[max(0, min(i, len(confidences) - 1))])

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "query": query,
                "seed_terms": seed_terms,
                "seed_count": len(seeds),
                "max_hops": cfg.max_hops,
                "max_nodes": cfg.max_nodes,
                "max_edges": cfg.max_edges,
                "min_confidence": cfg.min_confidence,
                "require_evidence": cfg.require_evidence,
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "edge_types": dict(etypes),
                    "evidence_rate": evidence_count / max(1, len(edges)),
                    "confidence_p50": q(0.50),
                    "confidence_p90": q(0.90),
                },
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        }
