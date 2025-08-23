# stephanie/utils/similarity_utils.py
"""
similarity_utils.py
-------------------
Utility functions for computing similarity scores between text items
based on their embedding vectors.

This module is primarily used by scorers (e.g., ProximityScorer) to
analyze semantic relationships between hypotheses, goals, and other
knowledge units in Stephanie.

Functions:
    - compute_similarity_matrix: Given a list of texts, computes
      pairwise cosine similarity scores using embeddings retrieved
      from memory.

Notes:
    - Embeddings are retrieved from `memory.embedding.get_or_create`.
    - Any text without an embedding is skipped, with a log entry emitted.
    - Returns a sorted list of similarity tuples, highest similarity first.
"""

import itertools
import numpy as np


def compute_similarity_matrix(
    texts: list[str], memory, logger
) -> list[tuple[str, str, float]]:
    """
    Compute pairwise cosine similarity between embeddings of input texts.

    Args:
        texts (list[str]): List of text strings to compare.
        memory: Stephanie memory object with an embedding interface.
                Expected to have `memory.embedding.get_or_create(text)` 
                which returns a vector or None.
        logger: Logger used to capture warnings or missing embeddings.

    Returns:
        list[tuple[str, str, float]]:
            A list of tuples of the form (text1, text2, similarity),
            sorted in descending order of similarity score.
    """
    vectors = []      # store embedding vectors
    valid_texts = []  # store only those texts that had valid embeddings

    # 1. Generate embeddings for all texts
    for h in texts:
        vec = memory.embedding.get_or_create(h)
        if vec is None:
            # Log missing embeddings but continue gracefully
            logger.log("MissingEmbedding", {"texts_snippet": h[:60]})
            continue
        vectors.append(vec)
        valid_texts.append(h)

    similarities = []

    # 2. Compute cosine similarity for every unique pair
    for i, j in itertools.combinations(range(len(valid_texts)), 2):
        h1 = valid_texts[i]
        h2 = valid_texts[j]
        sim = float(
            np.dot(vectors[i], vectors[j])
            / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
        )
        similarities.append((h1, h2, sim))

    # 3. Sort results by similarity score (highest first)
    return sorted(similarities, key=lambda x: x[2], reverse=True)
