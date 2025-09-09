# stephanie/memory/hnet_embedding_store.py
from stephanie.memory.base_embedding_store import BaseEmbeddingStore


class HNetEmbeddingStore(BaseEmbeddingStore):
    def __init__(self, cfg, memory, logger, cache_size=10000):
        super().__init__(cfg, memory, logger=logger, table="hnet_embeddings", name="hnet", cache_size=cache_size)
