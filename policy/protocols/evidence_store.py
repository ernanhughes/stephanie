from typing import List, Protocol, Tuple

import numpy as np


class EvidenceStore(Protocol):
    """
    Generic supporting data interface.
    Dataset-agnostic.
    """

    def has(self, element_id: str) -> bool:
        ...

    def get_texts(self, element_ids: List[str]) -> List[str]:
        ...

    def get_texts_and_vectors(
        self,
        element_ids: List[str],
        model: str,
    ) -> Tuple[List[str], np.ndarray, List[str]]:
        """
        Returns:
            texts
            vectors
            missing_ids
        """
