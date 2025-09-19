# stephanie/memory/scorable_entity_store.py
from __future__ import annotations

from typing import List

from sqlalchemy.dialects.postgresql import insert as pg_insert

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.ner_retriever import NERRetrieverEmbedder
from stephanie.models.scorable_entity import ScorableEntityORM


class ScorableEntityStore(BaseSQLAlchemyStore):
    """
    Store for named entities linked to any Scorable
    (documents, plan_traces, prompts, etc.)
    """
    orm_model = ScorableEntityORM
    default_order_by = ScorableEntityORM.created_at.asc()

    def __init__(self, session_or_maker, memory, logger=None):
        super().__init__(session_or_maker, logger)
        self.memory = memory
        self.name = "scorable_entities"

        # Shared retriever (backed by memory.embedding)
        self.retriever = NERRetrieverEmbedder(memory=memory, logger=self.logger)

    def name(self) -> str:
        return self.name

    def insert(self, data: dict) -> ScorableEntityORM:
        """
        Insert or update a scorable entity.

        Expected dict keys:
            - scorable_id (str)
            - scorable_type (str)
            - entity_text (str)
            - entity_type (str)
            - start (int)
            - end (int)
            - similarity (float)
            - source_text (str)
        """
        def op(s):
            stmt = (
                pg_insert(ScorableEntityORM)
                .values(**data)
                .on_conflict_do_update(
                    index_elements=["scorable_id", "scorable_type", "entity_text"],
                    set_={
                        "entity_type": data.get("entity_type"),
                        "start": data.get("start"),
                        "end": data.get("end"),
                        "similarity": data.get("similarity"),
                        "source_text": data.get("source_text"),
                    },
                )
                .returning(ScorableEntityORM.id)
            )

            
            result = s.execute(stmt)
            inserted_id = result.scalar()

            if inserted_id and self.logger:
                self.logger.log("EntityUpserted", data)

            return (
                s.query(ScorableEntityORM)
                .filter_by(
                    scorable_id=data["scorable_id"],
                    scorable_type=data["scorable_type"],
                    entity_text=data["entity_text"],
                )
                .first()
            )
        return self._run(op)

    def get_by_scorable(self, scorable_id: str, scorable_type: str) -> List[ScorableEntityORM]:
        """Get all entities linked to a scorable."""
        def op(s):
            return (
                s.query(ScorableEntityORM)
                .filter_by(scorable_id=scorable_id, scorable_type=scorable_type)
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

    def index(self, scorables: List) -> int:
        """
        Detect and index entities for a list of scorables.
        Uses the retriever + embedding store, persists to DB, and updates Annoy.
        """
        count = 0
        for scorable in scorables:
            entities = self.retriever.entity_detector.detect_entities(scorable.text)
            for ent in entities:
                entity_text = ent["text"].strip()
                if not entity_text:
                    continue
                self.insert({
                    "scorable_id": str(scorable.id),
                    "scorable_type": scorable.target_type,
                    "entity_text": entity_text,
                    "entity_type": ent["type"],
                    "start": ent["start"],
                    "end": ent["end"],
                    "similarity": ent.get("score"),
                    "source_text": scorable.text[:100] + "..."
                })
                count += 1

        # Update retriever Annoy index too
        self.retriever.index_scorables(scorables)
        return count
