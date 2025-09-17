# stephanie/memory/hnet_embedding_store.py
from __future__ import annotations

from stephanie.memory.base_embedding_store import BaseEmbeddingStore
from stephanie.tools import get_embedding


class EmbeddingStore(BaseEmbeddingStore):
    def __init__(self, cfg, memory, logger, cache_size=10000):
        super().__init__(cfg, memory, logger=logger, table="ollama_embeddings", name="ollama", embed_fn=get_embedding, cache_size=cache_size)
