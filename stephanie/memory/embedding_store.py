# stephanie/memory/hnet_embedding_store.py
from stephanie.memory.base_embedding_store import BaseEmbeddingStore
from stephanie.tools import get_embedding


class EmbeddingStore(BaseEmbeddingStore):
    def __init__(self, cfg, conn, db, logger=None, cache_size=10000):
        super().__init__(cfg, conn, db, table="ollama_embeddings", name="ollama", embed_fn=get_embedding, logger=logger, cache_size=cache_size)
