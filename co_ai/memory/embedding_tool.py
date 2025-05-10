# co_ai/memory/embedding_tool.py
import numpy as np
import requests

def get_embedding(text: str, cfg):
    """
    Get an embedding from Ollama using the configured model.

    Args:
        text (str): The input text to embed.
        cfg (dict or omegaconf.DictConfig): Configuration containing 'model' and optionally 'endpoint'.

    Returns:
        list[float]: The embedding vector.
    """
    
    model = cfg.get("embeddings", {}).get("model", "mxbai-embed-large")
    endpoint = cfg.get("embeddings", {}).get("endpoint", "FIX http://localhost:11434/api/embeddings")
    print(f"cfg: {cfg}")
    print(f"Using model: {model} at endpoint: {endpoint}")
    response = requests.post(
        endpoint,
        json={"model": model, "prompt": text},
    )
    response.raise_for_status()
    return response.json().get("embedding")
