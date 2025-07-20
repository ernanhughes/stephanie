from stephanie.protocols.base import Protocol
from stephanie.tools.embedding_tool import MXBAIEmbedder
from stephanie.tools.hnet_embedder import StephanieHNetEmbedder


class HNetEmbeddingProtocol(Protocol):
    def __init__(self, base_embedder=None):
        self.base_embedder = base_embedder or MXBAIEmbedder()
        self.embedder = StephanieHNetEmbedder(self.base_embedder)

    def run(self, input_context: dict) -> dict:
        text = input_context.get("text", "")
        embedding = self.embedder.embed(text)
        return {
            "embedding": embedding,
            "protocol_used": "hnet",
            "embedding_dim": len(embedding),
            "chunk_count": len(self.embedder.chunker.chunk(text)),
            "source": "HNetEmbeddingProtocol"
        }