# stephanie/agents/maintenance/document_embedding_backfill.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.document import DocumentORM
from stephanie.memory.document_embedding_store import DocumentEmbeddingStore


class DocumentEmbeddingBackfillAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.store = DocumentEmbeddingStore(memory.session, logger=logger)
        self.document_type = cfg.get("document_type", "document")  # e.g., "document", "prompt"
        self.document_embedding_type = cfg.get("document_embedding_type", "document")  # e.g., "full", "summary"

    async def run(self, context: dict) -> dict:
        session = self.memory.session

        # Step 1: Find documents missing embedding_id
        documents = (
            session.query(DocumentORM) 
            .filter(DocumentORM.embedding_id == None)
            .all()
        )
        updated = 0

        for doc in documents:
            if self.document_embedding_type == "full":
                combined_text = f"{doc.title or ''}\n{doc.summary or ''}".strip()
            elif self.document_embedding_type == "summary":
                combined_text = doc.summary or ''

            if not combined_text:
                continue

            # Step 2: Generate embedding vector with backend
            embedding_vector = self.memory.embedding.get_or_create(combined_text)
            embedding_id = self.memory.embedding.get_id_for_text(combined_text)

            if embedding_id:
                # Step 3: Insert into document_embeddings store
                self.store.insert({
                    "document_id": str(doc.id),
                    "document_type": self.type,
                    "embedding_id": embedding_id,
                    "embedding_type": self.memory.embedding.backend_name,
                })

                # (Optional) keep shortcut on DocumentORM
                doc.embedding_id = embedding_id
                updated += 1

        session.commit()
        self.logger.log("DocumentEmbeddingsBackfilled", {"updated": updated})
        context["embedding_updates"] = updated
        return context
