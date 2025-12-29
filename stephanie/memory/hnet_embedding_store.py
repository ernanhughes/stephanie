# stephanie/memory/hnet_embedding_store.py
from __future__ import annotations

from stephanie.memory.base_embedding_store import BaseEmbeddingStore
from stephanie.orm.hnet_embedding import HNetEmbeddingORM
from stephanie.tools.hnet_embedding import get_embedding


class HNetEmbeddingStore(BaseEmbeddingStore):
    orm_model = HNetEmbeddingORM
    default_order_by = HNetEmbeddingORM.created_at.desc()
    def __init__(self, cfg, memory, logger, cache_size=10000):
        super().__init__(cfg, memory, logger=logger, table="hnet_embeddings", name="hnet", embed_fn=get_embedding, cache_size=cache_size)
