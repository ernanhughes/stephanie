# stephanie/embeddings/base_protocol.py
from abc import ABC, abstractmethod


class EmbeddingProtocol(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
