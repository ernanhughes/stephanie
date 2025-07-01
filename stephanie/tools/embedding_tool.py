# stephanie/tools/embedding_tool.py
from collections import OrderedDict

import requests


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
    endpoint = cfg.get("embeddings", {}).get("endpoint", "http://localhost:11434/api/embeddings")
    response = requests.post(
        endpoint,
        json={"model": model, "prompt": text},
    )
    response.raise_for_status()
    return response.json().get("embedding")
