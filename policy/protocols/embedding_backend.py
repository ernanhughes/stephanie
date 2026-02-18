from typing import List, Protocol, Tuple

import numpy as np


class EmbeddingBackend(Protocol):
    """
    Storage contract for embeddings.
    Does NOT compute embeddings.
    """

    def get(
        self,
        texts: List[str],
        model: str,
    ) -> Tuple[List[np.ndarray | None], List[int]]:
        """
        Retrieve embeddings.

        Returns:
            vecs: aligned to input order (None where missing)
            missing_indices: indices of texts not found
        """
        ...

    def put(
        self,
        texts: List[str],
        vecs: np.ndarray,
        model: str,
    ) -> None:
        """
        Persist embeddings.
        """
        ...
