# stephanie/memory/scorable_entity_store.py
from __future__ import annotations

from typing import List

from sqlalchemy.dialects.postgresql import insert as pg_insert

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.ner_retriever import NERRetrieverEmbedder
from stephanie.models.scorable_entity import ScorableEntityORM


# Key changes only — not a full file replacement.

class ScorableEntityStore(BaseSQLAlchemyStore):
    orm_model = ScorableEntityORM
    default_order_by = ScorableEntityORM.created_at.asc()

    def __init__(self, session_or_maker, memory, logger=None):
        super().__init__(session_or_maker, logger)
        self.memory = memory
        self.name = "scorable_entities"
        self.retriever = NERRetrieverEmbedder(memory=memory, logger=self.logger)

    @staticmethod
    def _normalize_entity_text(s: str) -> str:
        return " ".join((s or "").strip().split()).lower()

    def insert(self, data: dict) -> ScorableEntityORM:
        # augment fields
        data = dict(data)
        data.setdefault("ner_confidence", data.pop("similarity", None))
        data.setdefault("entity_text_norm", self._normalize_entity_text(data.get("entity_text","")))

        def op(s):
            stmt = (
                pg_insert(ScorableEntityORM)
                .values(**data)
                .on_conflict_do_update(
                    index_elements=["scorable_id", "scorable_type", "entity_text_norm"],
                    set_={
                        "entity_type": data.get("entity_type"),
                        "start": data.get("start"),
                        "end": data.get("end"),
                        "ner_confidence": data.get("ner_confidence"),
                        "source_text": data.get("source_text"),
                        # keep retrieval_similarity nullable here; populated during search
                    },
                )
                .returning(ScorableEntityORM.id)
            )
            rid = s.execute(stmt).scalar()
            if rid and self.logger: 
                self.logger.log("EntityUpserted", data)
            return (
                s.query(ScorableEntityORM)
                .filter_by(
                    scorable_id=data["scorable_id"],
                    scorable_type=data["scorable_type"],
                    entity_text_norm=data["entity_text_norm"],
                )
                .first()
            )
        return self._run(op)

    async def index(self, scorables: List) -> int:
        count = 0
        async def op(s):
            nonlocal count
            for sc in scorables:
                try:
                    ents = self.retriever.entity_detector.detect_entities(sc.text) or []
                    for e in ents[: self.memory.cfg.get("max_entities_per_scorable", 300)]:
                        # after entities = kg.detect_entities(text)
                        e_text = (e.get("text") or "").strip()
                        if not e_text: 
                            continue
                        self.memory.scorable_entities.insert({
                            "scorable_id": str(sc.id),
                            "scorable_type": sc.target_type,
                            "entity_text": e["text"],
                            "entity_type": e.get("type","UNKNOWN"),
                            "start": int(e.get("start",-1)),
                            "end": int(e.get("end",-1)),
                            "similarity": float(e.get("score", 0.0)),
                            "source_text": sc.text[:100] + "..."
                        })
                        # optionally publish to KG (same payload shape KG expects)
                        await self.memory.bus.publish("knowledge_graph.index_request", {
                            "scorable_id": str(sc.id),
                            "scorable_type": sc.target_type,
                            "text": sc.text,
                            "entities": ents,
                            "domains": sc.meta.get("domains", []),
                        })

                        count += 1
                except Exception as e:
                    self.logger.log("EntityIndexError", {"id": str(getattr(sc, "id", "?")), "err": str(e)})
            return count
        total = self._run(op)
        # Update ANN/HNSW index once per batch
        try:
            self.retriever.index_scorables(scorables)
        except Exception as e:
            if self.logger: self.logger.log("EntityANNIndexError", {"err": str(e)})
        return total

    # Convenience: get near chats for a given document scorable
    def related_chats_for_document(self, *, scorable_id: str, k: int = 20, domain: str | None = None) -> List[dict]:
        """
        1) Load entities for the document scorable.
        2) For each entity_text, query retriever.retrieve_entities(...).
        3) Map results back to conversation turns (via scorable metadata).
        4) Merge, de-dup, score by max calibrated_similarity.
        """
        ents = self.find(scorable_id, scorable_type="document")
        if not ents: return []

        hits = []
        for e in ents[:50]:  # cap per-doc
            results = self.retriever.retrieve_entities(
                query=e.entity_text,
                k=max(5, k // 2),
                domain=domain or getattr(e, "domain", None)
            )
            for r in results:
                # populate retrieval_similarity if you want to persist later
                r["entity"] = e.entity_text
                hits.append(r)

        # merge by (scorable_id, target_type)
        by_key = {}
        for h in hits:
            key = (str(h.get("scorable_id", h.get("id"))), h.get("scorable_type"))
            score = h.get("calibrated_similarity", h.get("similarity", 0.0))
            if key not in by_key or score > by_key[key]["score"]:
                by_key[key] = {
                    "scorable_id": key[0],
                    "target_type": key[1],
                    "best_entity_match": h.get("entity_text"),
                    "score": score,
                    "raw_similarity": h.get("raw_similarity", h.get("similarity")),
                    "source_snippet": h.get("source_text"),
                }

        # TODO: map scorable_id+target_type → actual chat turn objects using your memory APIs
        ranked = sorted(by_key.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:k]

    def find(self, scorable_id: str, scorable_type: str) -> List[ScorableEntityORM]:
        """Get all entities linked to a scorable."""
        def op(s):
            return (
                s.query(ScorableEntityORM)
                .filter_by(scorable_id=str(scorable_id), scorable_type=scorable_type)
                .all()
            )
        return self._run(op)

    def delete_by_scorable(self, scorable_id: str, scorable_type: str):
        """Delete all entities linked to a scorable."""
        def op(s):
            
            s.query(ScorableEntityORM).filter_by(
                scorable_id=scorable_id, scorable_type=scorable_type
            ).delete()
            if self.logger:
                self.logger.log("EntitiesDeleted", {
                    "scorable_id": scorable_id,
                    "scorable_type": scorable_type
                })
        return self._run(op)

