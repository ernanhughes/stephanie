# stephanie/services/subgraphs/seed_finder.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


class EmbeddingSeedFinder:
    """
    Responsible only for: query -> ranked seed node IDs (plus seed terms).
    """

    def __init__(
        self,
        *,
        search_entities_fn: Callable[[str, int], List[Tuple[str, float, Dict[str, Any]]]],
        detect_entities_fn: Callable[[str], List[Dict[str, Any]]],
        logger: Any = None,
    ) -> None:
        self.search_entities_fn = search_entities_fn
        self.detect_entities_fn = detect_entities_fn
        self.logger = logger

    def find_seeds(
        self,
        *,
        query: str,
        seed_k: int,
        per_entity_k: int,
        k_entities: int,
        min_seed_score: float,
        max_seeds: int,
    ) -> Dict[str, Any]:
        query = (query or "").strip()
        seed_terms: List[str] = []

        try:
            ents = self.detect_entities_fn(query)[: max(1, int(k_entities))]
            seed_terms = _dedupe_keep_order(
                [e.get("text", "").strip() for e in ents if e.get("text")]
            )
        except Exception as ex:
            if self.logger:
                self.logger.warning(f"EmbeddingSeedFinder: detect_entities failed: {ex}")

        scored: List[Tuple[str, float]] = []

        try:
            for node_id, score, _meta in self.search_entities_fn(query, k=int(seed_k)):
                if float(score) >= float(min_seed_score):
                    scored.append((str(node_id), float(score)))

            for term in seed_terms[: int(k_entities)]:
                for node_id, score, _meta in self.search_entities_fn(term, k=int(per_entity_k)):
                    if float(score) >= float(min_seed_score):
                        scored.append((str(node_id), float(score)))
        except Exception as ex:
            if self.logger:
                self.logger.warning(f"EmbeddingSeedFinder: search_entities failed: {ex}")

        scored.sort(key=lambda x: x[1], reverse=True)

        seeds: List[str] = []
        seen = set()
        for nid, _ in scored:
            if nid and nid not in seen:
                seeds.append(nid)
                seen.add(nid)
                if len(seeds) >= int(max_seeds):
                    break

        return {
            "seeds": seeds,
            "seed_terms": seed_terms,
            "seed_count": len(seeds),
        }
