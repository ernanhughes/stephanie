# stephanie/memory/hnet_embedding_store.py
from __future__ import annotations

from stephanie.memory.base_embedding_store import BaseEmbeddingStore
from stephanie.models.embedding import EmbeddingORM
from stephanie.tools.ollama_embedding import get_embedding


class EmbeddingStore(BaseEmbeddingStore):
    orm_model = EmbeddingORM
    default_order_by = EmbeddingORM.created_at.desc()

    def __init__(self, cfg, memory, logger, cache_size=10000):
        super().__init__(cfg, memory, logger=logger, table="ollama_embeddings", name="ollama", embed_fn=get_embedding, cache_size=cache_size)
