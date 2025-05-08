# ai_co_scientist/memory/embedding_tool.py
import requests
import numpy as np

# Uses Ollama locally to generate embeddings via an embedding model like "nomic-embed-text"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"


def get_embedding(text: str):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
    )
    response.raise_for_status()
    return response.json().get("embedding")
