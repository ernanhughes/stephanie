# stephanie/services/knowledge_graph/graph_indexer.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from stephanie.utils.hash_utils import hash_text
from stephanie.utils.text_utils import safe_slice, sentences

log = logging.getLogger(__name__)

class GraphIndexer:
    def __init__(self, *, kg_service: Any, 
                 detect_entities_fn: Any,
                 fetch_text_fn: Any,
                 logger: Any):
        from stephanie.services.knowledge_graph_service import \
            KnowledgeGraphService
        self.kg: KnowledgeGraphService = kg_service
        self.detect_entities_fn = detect_entities_fn
        self.fetch_text_fn = fetch_text_fn
        self.logger = logger

    def _normalize_text(self, t: str) -> str:
        return (t or "").replace("\u200b", "").strip()

    def _locate_sentence_ix(self, sent_spans, start: int, end: int) -> int:
        for i, (s, e) in enumerate(sent_spans):
            if not (end <= s or start >= e):
                return i
        return -1

    async def _fetch_text_for_indexing(self, scorable_id: str, payload_text: Optional[str]) -> Optional[str]:
        # 1) try service hook if present
        try:
            fn = getattr(self.kg, "fetch_text_for_scorable", None)
            if callable(fn):
                t = fn(scorable_id)
                if t:
                    return t
        except Exception as ex:
            log.warning(f"GraphIndexer: fetch_text_for_scorable failed: {ex}")

        # 2) legacy: if service has scorable_store and a getter
        try:
            store = getattr(self.kg, "scorable_store", None)
            if store and hasattr(store, "get_by_scorable_id"):
                row = store.get_by_scorable_id(scorable_id)
                t = getattr(row, "text", None)
                if t:
                    return t
        except Exception as ex:
            log.warning(f"GraphIndexer: scorable_store lookup failed: {ex}")

        # 3) fallback
        return payload_text

    def _mention_id(self, scorable_id: str, ent: Dict[str, Any]) -> str:
        return f"{scorable_id}:{ent['type']}:{ent['start']}-{ent['end']}"

    def _repair_entities(
        self,
        *,
        entities: List[Dict[str, Any]],
        text: str,
        sent_spans,
        doc_hash: str,
    ) -> List[Dict[str, Any]]:
        fixed: List[Dict[str, Any]] = []
        for ent in (entities or []):
            etype = (ent.get("type") or "UNKNOWN").strip() or "UNKNOWN"
            surface = ent.get("text")

            try:
                start = int(ent.get("start", -1))
                end = int(ent.get("end", -1))
            except Exception:
                start, end = -1, -1

            etype = (ent.get("type") or "UNKNOWN").strip() or "UNKNOWN"
            surface = ent.get("text")

            # repair offsets if needed using surface search
            if start < 0 or end <= start or end > len(text):
                if surface:
                    pos = text.find(surface)
                    if pos >= 0:
                        start, end = pos, pos + len(surface)
                    else:
                        continue
                else:
                    continue

            derived = safe_slice(text, start, end)
            if not surface or surface != derived:
                surface = derived

            sentence_ix = self._locate_sentence_ix(sent_spans, start, end)

            if 0 <= sentence_ix < len(sent_spans):
                s0, s1 = sent_spans[sentence_ix]
                context = text[s0:s1]
            else:
                context = safe_slice(text, max(0, start - 60), min(len(text), end + 60))

            fixed.append(
                {
                    **ent,
                    "start": start,
                    "end": end,
                    "type": etype,
                    "text": surface,
                    "sentence_ix": sentence_ix,
                    "context": context,
                    "doc_hash": doc_hash,
                }
            )
        return fixed

    def _cooccur_edges(
        self,
        *,
        scorable_id: str,
        scorable_type: str,
        doc_hash: str,
        fixed_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_sent: Dict[int, List[Dict[str, Any]]] = {}
        for e in fixed_entities:
            by_sent.setdefault(int(e.get("sentence_ix", -1)), []).append(e)

        rels: List[Dict[str, Any]] = []
        for s_ix, ents in by_sent.items():
            if s_ix < 0 or len(ents) < 2:
                continue
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    a, b = ents[i], ents[j]
                    rels.append(
                        {
                            "source": self._mention_id(scorable_id, a),
                            "target": self._mention_id(scorable_id, b),
                            "type": "CO_OCCURS_IN_SENTENCE",
                            "confidence": 1.0,
                            "doc_hash": doc_hash,
                            "sentence_ix": s_ix,
                            "scorable_id": scorable_id,
                            "scorable_type": scorable_type,
                            "evidence_type": "sentence_cooccurrence",
                        }
                    )
        return rels



    def _build_relationships(
        self,
        *,
        fixed_entities: List[Dict[str, Any]],
        domains: List[Dict[str, Any]],
        scorable_id: str,
        scorable_type: str,
        doc_hash: str,
    ) -> List[Dict[str, Any]]:
        rels: List[Dict[str, Any]] = []

        # 1) typed rels (delegate to service if present)
        try:
            fn = getattr(self.kg, "_build_relationships", None)
            if callable(fn):
                # expects (entities, domains, scorable_id) in your service
                typed = fn(fixed_entities, domains, scorable_id) or []
                # normalize keys to {source,target,type,confidence}
                for r in typed:
                    rels.append(
                        {
                            "source": r.get("source") or r.get("source_id"),
                            "target": r.get("target") or r.get("target_id"),
                            "type": r.get("type") or r.get("rel_type") or "RELATED",
                            "confidence": float(r.get("confidence", 0.0) or 0.0),
                            "properties": r.get("properties") or {},
                        }
                    )
        except Exception as ex:
            log.warning(f"GraphIndexer: typed _build_relationships failed: {ex}")

        # 2) sentence co-occurrence (cheap signal)
        by_sent: Dict[int, List[Dict[str, Any]]] = {}
        for e in fixed_entities:
            by_sent.setdefault(int(e.get("sentence_ix", -1)), []).append(e)

        for s_ix, ents in by_sent.items():
            if s_ix < 0 or len(ents) < 2:
                continue
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    a, b = ents[i], ents[j]
                    rels.append(
                        {
                            "source": f"{scorable_id}:{a['type']}:{a['start']}-{a['end']}",
                            "target": f"{scorable_id}:{b['type']}:{b['start']}-{b['end']}",
                            "type": "CO_OCCURS_IN_SENTENCE",
                            "confidence": 1.0,
                            "properties": {
                                "doc_hash": doc_hash,
                                "sentence_ix": s_ix,
                                "scorable_id": scorable_id,
                                "scorable_type": scorable_type,
                                "evidence_type": "sentence_cooccurrence",
                            },
                        }
                    )

        return rels

    def _mention_id(self, scorable_id: str, ent: Dict[str, Any]) -> str:
        return f"{scorable_id}:{ent['type']}:{ent['start']}-{ent['end']}"

    async def handle_index_request(self, payload: Dict[str, Any]) -> None:
        scorable_id = payload["scorable_id"]
        scorable_type = payload.get("scorable_type", "unknown")
        entities = payload.get("entities") or []
        domains = payload.get("domains") or []

        text = await self.fetch_text_fn(scorable_id, payload.get("text"), payload)
        if not text:
            await self.kg.publish("knowledge_graph.index_failed", {"scorable_id": scorable_id, "error": "Missing text"})
            return

        text = self._normalize_text(text)
        doc_hash = hash_text(text)
        sent_spans = sentences(text)

        fixed_entities = self._repair_entities(entities=entities, text=text, sent_spans=sent_spans, doc_hash=doc_hash)
        if not entities:
            try:
                entities = self.detect_entities_fn(text) or []
            except Exception:
                log.warning("GraphIndexer: detect_entities_fn failed", exc_info=True)

        # upsert mention + canonical + MENTIONS edge
        for ent in fixed_entities:
            mention_id = self._mention_id(scorable_id, ent)

            # mention node
            self.kg.upsert_node(
                node_id=mention_id,
                properties={
                    "type": "entity_mention",
                    "text": ent["text"],
                    "entity_type": ent.get("type", "UNKNOWN"),
                    "scorable_id": scorable_id,
                    "scorable_type": scorable_type,
                    "doc_hash": doc_hash,
                    "sentence_ix": ent.get("sentence_ix"),
                    "context": ent.get("context"),
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                    "domains": [d.get("domain") for d in domains if isinstance(d, dict)],
                },
            )

            canon_id = self.kg._canonical_entity_id(ent)
            self.kg.upsert_node(
                node_id=canon_id,
                properties={
                    "type": "canonical_entity",
                    "text": ent["text"],
                    "entity_type": ent.get("type", "UNKNOWN"),
                    "domains": [d.get("domain") for d in domains if isinstance(d, dict)],
                },
            )

            self.kg.upsert_edge(
                source_id=mention_id,
                target_id=canon_id,
                rel_type="MENTIONS",
                properties={
                    "confidence": 0.95,
                    "doc_hash": doc_hash,
                    "sentence_ix": ent.get("sentence_ix"),
                    "scorable_id": scorable_id,
                    "scorable_type": scorable_type,
                    "evidence_type": "entity_mention_link",
                },
            )

        # relationships (typed + co-occurrence)
        rels = self._build_relationships(
            fixed_entities=fixed_entities,
            domains=domains,
            scorable_id=scorable_id,
            scorable_type=scorable_type,
            doc_hash=doc_hash,
        )
        for rel in rels:
            self.kg.upsert_edge(
                source_id=rel["source"],
                target_id=rel["target"],
                rel_type=rel["type"],
                properties={
                    "confidence": float(rel.get("confidence", 1.0)),
                    **(rel.get("properties") or {}),
                },
            )

        await self.kg.publish(
            "knowledge_graph.index_complete",
            {
                "scorable_id": scorable_id,
                "node_count": len(fixed_entities),
                "relationship_count": max(0, len(fixed_entities) - 1),
                "doc_hash": doc_hash,
            },
        )
