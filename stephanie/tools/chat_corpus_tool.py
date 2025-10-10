# stephanie/tools/chat_corpus_tool.py
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from stephanie.scoring.scorable import ScorableType

_logger = logging.getLogger(__name__)


def build_chat_corpus_tool(
    memory: Any,
    container: Any,
    cfg: Optional[Dict[str, Any]] = None,
):
    """
    Factory: returns a single callable `chat_corpus(query_text, **kwargs)`.

    The callable is PURE READ-ONLY:
      - Pulls existing chat turns and per-turn annotations (entities/domains)
      - Scores them vs. the provided text
      - Returns ranked items with semantic/entity/domain scores + reasons

    Dependencies (read-only):
      - memory.embedding
      - memory.scorable_entities
      - memory.scorable_domains
      - container.get("knowledge_graph")._entity_detector (optional, for query-side NER)

    Usage:
        chat_corpus = build_chat_corpus_tool(memory=memory, container=container, logger=logger)
        result = chat_corpus("section text...", k=60, mode="hybrid")
    """
    cfg = cfg or {}

    def _norm(x) -> str:
        if x is None:
            s = ""
        elif isinstance(x, str):
            s = x
        elif isinstance(x, dict):
            s = x.get("domain") or x.get("name") or x.get("label") or ""
        elif isinstance(x, (tuple, list)) and x and isinstance(x[0], str):
            s = x[0]
        else:
            s = str(x)
        return " ".join(s.strip().lower().split())

    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))

    def _overlap_ratio(a, b) -> float:
        """
        Jaccard-like overlap on *names*, robust to list/tuple/set inputs.
        Returns |A ∩ B| / max(1, |A ∪ B|).
        """
        try:
            A = set(a) if not isinstance(a, set) else a
            B = set(b) if not isinstance(b, set) else b
        except Exception:
            A, B = set(), set()
        if not A and not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))

    def _entities_for_text(text: str) -> List[str]:
        try:
            kg = container.get("knowledge_graph")
            det = getattr(kg, "_entity_detector", None)
            ents = ents = det.detect_entities(text) or []
            if ents:
                return sorted(
                    {_norm(e.get("text")) for e in ents if e.get("text")}
                )
        except Exception as e:
            _logger.warning("Entity detection failed: %s", e)
        toks = re.findall(r"\b[A-Z][A-Za-z0-9\-\_]{2,}\b", text or "")
        return sorted({_norm(t) for t in toks})

    def _domains_for_text(
        text: str,
        *,
        top_k: int = 5,
        min_value: float = 0.10,
        context: dict | None = None,
    ) -> List[str]:
        """
        Return a normalized list of domain names (strings). Never tuples.
        Mirrors ScorableClassifier.classify(text) -> List[Tuple[str, float]].
        """
        if not isinstance(text, str) or not text.strip():
            return []
        # Try KG classifier
        try:
            kg = container.get("knowledge_graph")
            clf = getattr(kg, "_classifier", None)
            if clf:
                results: List[Tuple[str, float]] = (
                    clf.classify(
                        text, top_k=top_k, min_value=min_value, context=context
                    )
                    or []
                )
                return [
                    _norm(name)
                    for (name, _score) in results
                    if isinstance(name, str)
                ]
        except Exception as e:
            _logger.warning("domain classify failed: %s", e)

        # Fallback: very conservative heuristics -> empty (or add your own simple rules)
        return []

    def _embedding_candidates(text: str, k: int) -> List[Dict[str, Any]]:
        emb = getattr(memory, "embedding", None)
        if not emb:
            _logger.warning("Embedding index unavailable; no candidates.")
            return []
        try:
            cands = emb.search_related_scorables(
                text,
                ScorableType.CONVERSATION_TURN,
                include_ner=True,
                top_k=int(k),
            )
        except Exception as e:
            _logger.warning("Embedding search failed: %s", e)
            return []
        out = []
        for c in cands or []:
            out.append(
                {
                    "id": c.get("id") or c.get("scorable_id"),
                    "thread_id": (c.get("meta") or {}).get("thread_id"),
                    "role": (c.get("meta") or {}).get("role") or "assistant",
                    "text": c.get("text") or "",
                    "score": float(c.get("score") or 0.0),
                    "meta": c.get("meta") or {},
                }
            )
        return out

    def chat_corpus(
        query_text: str,
        *,
        k: int = 60,
        weights: Optional[Dict[str, float]] = None,
        candidate_multiplier: int = 3,
        include_text: bool = True,
        tags: Optional[List[str]] = None,   # NEW
    ) -> Dict[str, Any]:
        qt = (query_text or "").strip()
        if not qt:
            return {"items": [], "meta": {"k": k, "tags": tags or []}}

        w = {"semantic": 0.6, "entity": 0.25, "domain": 0.15}
        if weights:
            w.update(weights)

        # 1) candidates from embedding index
        cands = _embedding_candidates(qt, k * max(2, candidate_multiplier))
        if not cands:
            return {"items": [], "meta": {"k": k, "tags": tags or []}}

        cand_ids = [
            c.get("id") or c.get("scorable_id")
            for c in cands
            if (c.get("id") or c.get("scorable_id"))
        ]

        # 2) fetch snapshot for candidates
        snapshot = memory.chats.list_turns_by_ids_with_texts(cand_ids)

        # 2b) optional filter by conversation tags
        if tags:
            tags_set = {t.strip().lower() for t in tags}
            snapshot = [
                s for s in snapshot
                if tags_set.intersection(set(map(str.lower, (s.get("tags") or []))))
            ]

        snap_by_id = {int(s["id"]): s for s in snapshot}

        # 3) query-side features
        q_ents = _entities_for_text(qt)
        q_doms = _domains_for_text(qt)
        q_ent_set, q_dom_set = set(q_ents), set(q_doms)

        # 4) score + merge
        items: List[Dict[str, Any]] = []
        for c in cands:
            cid = int(c.get("id") or c.get("scorable_id"))
            sem = float(c.get("score") or 0.0)

            snap = snap_by_id.get(cid)  # may be None if not found
            # text for scoring fallbacks
            ctext = (snap and snap.get("assistant_text")) or (
                c.get("text") or ""
            )

            # candidate-side features (prefer stored; fallback to recompute when missing)
            c_dom_list = (snap and snap.get("domains")) or []
            c_ent_list = (snap and snap.get("ner")) or _entities_for_text(
                ctext
            )

            ent_overlap = _jaccard(
                q_ent_set, set(_norm(x) for x in c_ent_list)
            )
            dom_overlap = _overlap_ratio(
                [_norm(x) for x in q_doms], [_norm(x) for x in c_dom_list]
            )

            combined = (
                w["semantic"] * sem
                + w["entity"] * ent_overlap
                + w["domain"] * dom_overlap
            )

            reasons = []
            if sem >= 0.75:
                reasons.append("high semantic similarity")
            elif sem >= 0.55:
                reasons.append("moderate semantic similarity")
            inter_doms = sorted(set(_norm(x) for x in c_dom_list) & q_dom_set)
            inter_ents = sorted(set(_norm(x) for x in c_ent_list) & q_ent_set)
            if inter_doms:
                reasons.append(f"matched domains: {', '.join(inter_doms)}")
            if inter_ents:
                reasons.append(
                    f"shared entities: {', '.join(list(inter_ents)[:8])}"
                )

            # Base payload = EXACT FIELD SET from list_turns_for_conversation_with_texts
            base = snap or {
                "id": cid,
                "conversation_id": c.get("thread_id"),
                "order_index": int(
                    (c.get("meta") or {}).get("order_index") or 0
                ),
                "star": int((c.get("meta") or {}).get("star") or 0),
                "user_text": (c.get("meta") or {}).get("user_text") or "",
                "assistant_text": ctext or "",
                "ner": c_ent_list,
                "domains": c_dom_list,
                "goal_text": (c.get("meta") or {}).get("goal_text") or "",
                "ai_score": (c.get("meta") or {}).get("ai_score"),
                "ai_rationale": (c.get("meta") or {}).get("ai_rationale")
                or "",
                "scorable_id": str(cid),
                "scorable_type": ScorableType.CONVERSATION_TURN,
            }

            # Add your scoring ornaments (non-breaking)
            base = dict(base)  # copy
            base["scores"] = {
                "semantic": round(sem, 4),
                "entity_overlap": round(ent_overlap, 4),
                "domain_overlap": round(dom_overlap, 4),
                "combined": round(float(combined), 4),
            }
            base["matched_domains"] = inter_doms
            base["matched_entities"] = list(inter_ents)[:50]
            base["reasons"] = reasons

            if not include_text:
                # keep fields but blank out big text fields if you want to be lean
                base["assistant_text"] = ""
                base["user_text"] = ""

            items.append(base)

        items.sort(key=lambda r: r["scores"]["combined"], reverse=True)
        return {
            "items": items[:k],
            "meta": {
                "k": k,
                "weights": w,
                "query_domains": q_doms,
                "query_entities": q_ents[:50],
            },
        }

    return chat_corpus
