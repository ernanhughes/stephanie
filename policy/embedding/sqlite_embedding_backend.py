
from policy.embedding.embedding_store import EmbeddingStore


class SQLiteEmbeddingBackend:
    def __init__(self, db_path: str):
        self.store = EmbeddingStore(db_path)

    def get(self, texts, model):
        return self.store.get(texts, model)

    def put(self, texts, vecs, model):
        self.store.put(texts, vecs, model)

