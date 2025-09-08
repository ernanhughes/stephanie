# stephanie/memory/scorable_entity_store.py
import logging
from typing import List

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from stephanie.models.scorable_entity import ScorableEntityORM
from stephanie.scoring.model.ner_retriever import NERRetrieverEmbedder

logger = logging.getLogger(__name__)


class ScorableEntityStore:
    """
    Store for named entities linked to any Scorable
    (documents, plan_traces, prompts, etc.)
    """

    def __init__(self, session: Session, memory, logger=None):
        self.session = session
        self.memory = memory
        self.logger = logger or logging.getLogger(__name__)
        self.name = "scorable_entities"

        # Shared retriever (backed by memory.embedding)
        self.retriever = NERRetrieverEmbedder(memory)

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

        result = self.session.execute(stmt)
        inserted_id = result.scalar()
        self.session.commit()

        if inserted_id:
            self.logger.info(f"EntityUpserted: {data}")

        return (
            self.session.query(ScorableEntityORM)
            .filter_by(
                scorable_id=data["scorable_id"],
                scorable_type=data["scorable_type"],
                entity_text=data["entity_text"],
            )
            .first()
        )

    def get_by_scorable(self, scorable_id: str, scorable_type: str) -> List[ScorableEntityORM]:
        """
        Get all entities linked to a scorable.
        """
        return (
            self.session.query(ScorableEntityORM)
            .filter_by(scorable_id=scorable_id, scorable_type=scorable_type)
            .all()
        )

    def delete_by_scorable(self, scorable_id: str, scorable_type: str):
        """
        Delete all entities linked to a scorable.
        """
        self.session.query(ScorableEntityORM).filter_by(
            scorable_id=scorable_id, scorable_type=scorable_type
        ).delete()
        self.session.commit()
        self.logger.info(f"EntitiesDeleted: {scorable_type}:{scorable_id}")

    def index(self, scorables: List) -> int:
        """
        Detect and index entities for a list of scorables.
        Uses the retriever + embedding store, persists to DB, and updates Annoy.
        """
        count = 0
        for scorable in scorables:
            entities = self.retriever.entity_detector.detect_entities(scorable.text)
            for start, end, entity_type in entities:
                entity_text = scorable.text[start:end].strip()
                if not entity_text:
                    continue
                self.insert({
                    "scorable_id": str(scorable.id),
                    "scorable_type": scorable.target_type,
                    "entity_text": entity_text,
                    "entity_type": entity_type,
                    "start": start,
                    "end": end,
                    "similarity": None,
                    "source_text": scorable.text[:100] + "..."
                })
                count += 1

        # Update retriever Annoy index too
        self.retriever.index_scorables(scorables)
        return count
