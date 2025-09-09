# stephanie/memory/hf_embedding_store.py
from stephanie.memory.base_embedding_store import BaseEmbeddingStore


class HuggingFaceEmbeddingStore(BaseEmbeddingStore):
    def __init__(self, cfg, memory, logger, cache_size=10000):
        super().__init__(cfg, memory, logger=logger, table="hf_embeddings", name="hf",  cache_size=cache_size)
