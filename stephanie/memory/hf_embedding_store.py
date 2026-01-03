# stephanie/memory/hf_embedding_store.py
from __future__ import annotations

from stephanie.memory.base_embedding_store import BaseEmbeddingStore
from stephanie.orm.huggingface_embedding import HuggingfaceEmbeddingORM
from stephanie.tools.hf_embedding import get_embedding


class HuggingFaceEmbeddingStore(BaseEmbeddingStore):
    orm_model = HuggingfaceEmbeddingORM
    default_order_by = HuggingfaceEmbeddingORM.created_at.desc()
    def __init__(self, cfg, memory, logger, cache_size=10000):
        super().__init__(cfg, memory, logger=logger, table="hf_embeddings", name="hf", embed_fn=get_embedding, cache_size=cache_size)
