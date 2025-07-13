# stephanie/tools/cos_sim_tool.py
from typing import List, Tuple

import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2 + 1e-8)  # Avoid division by zero


def get_top_k_similar(
    query: str, documents: List[str], memory, top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Compute similarity between query and each document, return top_k most similar.

    Args:
        query: The input query text.
        documents: A list of document strings.
        get_embedding: Callable that takes a string and returns a vector (np.ndarray).
        top_k: Number of top results to return.

    Returns:
        List of (document, similarity_score) tuples.
    """
    query_vec = memory.embedding.get_or_create(query)
    doc_vecs = [memory.embedding.get_or_create(doc) for doc in documents]

    similarities = [cosine_similarity(query_vec, vec) for vec in doc_vecs]
    scored = list(zip(documents, similarities))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]
