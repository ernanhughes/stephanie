# stephanie/agents/maintenance/document_embedding_backfill.py


from stephanie.agents.base_agent import BaseAgent


class DocumentEmbeddingBackfillAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        session = self.memory.session

        from stephanie.models.document import DocumentORM

        # Step 1: Find documents missing embedding_id
        documents = (
            session.query(DocumentORM).filter(DocumentORM.embedding_id == None).all()
        )
        updated = 0

        for doc in documents:
            combined_text = f"{doc.title or ''}\n{doc.summary or ''}".strip()
            if not combined_text:
                continue

            # Step 2: Generate or retrieve embedding
            embedding_vector = self.memory.embedding.get_or_create(combined_text)
            embedding_id = self.memory.embedding.get_id_for_text(combined_text)

            if embedding_id:
                doc.embedding_id = embedding_id
                updated += 1

        session.commit()
        self.logger.log("DocumentEmbeddingsBackfilled", {"updated": updated})
        context["embedding_updates"] = updated
        return context
