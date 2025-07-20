from typing import List

from stephanie.protocols.embedding.base import EmbeddingProtocol
from stephanie.tools.embedding_tool import get_embedding


class StandardEmbedderProtocol(EmbeddingProtocol):
    def __init__(self, cfg:dict):
        self.model_name = cfg.get("model", "mxbai-embed-large")
        self.endpoint = cfg.get("endpoint", "http://localhost:11434/api/embeddings")

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text using Ollama or similar endpoint.
        """
        return get_embedding(text, {"model": self.model_name, "endpoint": self.endpoint})

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        """
        return [self.embed(text) for text in texts]

    def get_embedding_dim(self, text: str = "test") -> int:
        """
        Utility to detect embedding dimension.
        """
        sample = self.embed(text)
        return len(sample)

    def __repr__(self):
        return f"<StandardEmbedderProtocol model={self.model_name}>"