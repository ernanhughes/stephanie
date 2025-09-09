# stephanie/memory/hnet_embedding_store.py
from stephanie.memory.base_embedding_store import BaseEmbeddingStore


class EmbeddingStore(BaseEmbeddingStore):
    def __init__(self, cfg, memory, logger, cache_size=10000):
        super().__init__(cfg, memory, table="ollama_embeddings", name="ollama", logger=logger, cache_size=cache_size)
