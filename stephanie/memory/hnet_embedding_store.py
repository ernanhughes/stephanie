# stephanie/memory/hnet_embedding_store.py
from stephanie.memory.base_embedding_store import BaseEmbeddingStore
from stephanie.tools.hnet_embedder import get_embedding


class HNetEmbeddingStore(BaseEmbeddingStore):
    def __init__(self, cfg, conn, db, logger=None, cache_size=10000):
        super().__init__(cfg, conn, db, table="hnet_embeddings", name="hnet", embed_fn=get_embedding, logger=logger, cache_size=cache_size)
