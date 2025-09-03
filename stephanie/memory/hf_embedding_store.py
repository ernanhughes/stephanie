# stephanie/memory/hf_embedding_store.py
from stephanie.memory.base_embedding_store import BaseEmbeddingStore
from stephanie.tools.hf_embedding import get_embedding


class HuggingFaceEmbeddingStore(BaseEmbeddingStore):
    def __init__(self, cfg, conn, db, logger=None, cache_size=10000):
        super().__init__(cfg, conn, db, table="hf_embeddings", name="hf", embed_fn=get_embedding, logger=logger, cache_size=cache_size)
