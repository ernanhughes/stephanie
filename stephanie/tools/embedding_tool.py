# stephanie/tools/embedding_tool.py
from collections import OrderedDict

import requests

from stephanie.protocols.embedding.base import EmbeddingProtocol


# Simple in-memory LRU cache
class EmbeddingCache:
    def __init__(self, max_size=10000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            # Move to the end to mark as recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove least recently used item


embedding_cache = EmbeddingCache(max_size=10000)


class MXBAIEmbedder(EmbeddingProtocol):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dim = cfg.get("dim", 1024)
        self.hdim = self.dim / 2

    def embed(self, text: str) -> list[float]:
        return get_embedding(text, self.cfg)

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

def get_embedding(text: str, cfg):
    """
    Get an embedding from Ollama using the configured model.

    Args:
        text (str): The input text to embed.
        cfg (dict)): Configuration containing 'model' and optionally 'endpoint'.

    Returns:
        list[float]: The embedding vector.
    """
    cached = embedding_cache.get(text)
    if cached is not None:
        print("🔁 Using cached embedding")
        return cached

    model = cfg.get("embeddings", {}).get("model", "mxbai-embed-large")
    endpoint = cfg.get("embeddings", {}).get(
        "endpoint", "http://localhost:11434/api/embeddings"
    )
    response = requests.post(
        endpoint,
        json={"model": model, "prompt": text},
    )
    response.raise_for_status()
    return response.json().get("embedding")
