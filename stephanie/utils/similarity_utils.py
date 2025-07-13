# stephanie/utils/similarity_utils.py
# utils/similarity_utils.py
import itertools

import numpy as np


def compute_similarity_matrix(
    hypotheses: list[str], memory, logger
) -> list[tuple[str, str, float]]:
    vectors = []
    valid_hypotheses = []
    for h in hypotheses:
        vec = memory.embedding.get_or_create(h)
        if vec is None:
            logger.log("MissingEmbedding", {"hypothesis_snippet": h[:60]})
            continue
        vectors.append(vec)
        valid_hypotheses.append(h)

    similarities = []
    for i, j in itertools.combinations(range(len(valid_hypotheses)), 2):
        h1 = valid_hypotheses[i]
        h2 = valid_hypotheses[j]
        sim = float(
            np.dot(vectors[i], vectors[j])
            / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
        )
        similarities.append((h1, h2, sim))

    return sorted(similarities, key=lambda x: x[2], reverse=True)
