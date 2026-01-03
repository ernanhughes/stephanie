# stephanie/tools/similar_tool.py
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = float(np.dot(vec1, vec2))
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    return dot / (norm1 * norm2 + 1e-8)  # Avoid division by zero


def get_top_k_similar(
    query: str,
    documents: Sequence[str],
    *,
    embed: Callable[[str], np.ndarray],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Compute similarity between `query` and each item in `documents`
    using the provided `embed(text) -> vector` function.

    Returns:
        List of (document, similarity_score) sorted by similarity DESC.
    """
    if not documents:
        return []

    query_vec = embed(query)
    doc_vecs = [embed(doc) for doc in documents]

    similarities = [cosine_similarity(query_vec, vec) for vec in doc_vecs]
    scored = list(zip(documents, similarities))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def get_top_k_similar_with_memory(
    query: str,
    documents: Sequence[str],
    memory,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Backwards-compatible helper that uses Stephanie's embedding store.

    `memory.embedding` should be a BaseEmbeddingStore-like object
    exposing `.get_or_create(text) -> np.ndarray`.
    """
    embed = memory.embedding.get_or_create
    return get_top_k_similar(query, documents, embed=embed, top_k=top_k)


def pairwise_similarity_matrix(
    texts: Sequence[str],
    *,
    embed: Callable[[str], np.ndarray],
) -> np.ndarray:
    """
    Optional helper: build an N x N cosine similarity matrix for a set of texts.

    This will be handy when you want to build a section-to-section similarity
    graph inside a single paper or across a small cluster of papers.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    vecs = [embed(t) for t in texts]
    mat = np.stack(vecs).astype(np.float32)
    # Normalize row-wise
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    mat_normed = mat / norms
    return np.matmul(mat_normed, mat_normed.T)
