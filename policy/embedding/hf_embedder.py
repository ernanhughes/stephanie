from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from policy.protocols.embedding_backend import EmbeddingBackend


class HFEmbedder:
    """
    HuggingFace embedder that delegates storage
    to an EmbeddingBackend.
    """

    name: str = "HFEmbedder"

    def __init__(
        self,
        model_name: str,
        backend: EmbeddingBackend,
    ):
        self.model_name = model_name
        self.backend = backend
        self.model = SentenceTransformer(model_name)

        import warnings
        warnings.filterwarnings(
            "ignore",
            message=".*embeddings.position_ids.*"
        )

        # Simple in-memory cache (hashable key)
        self._memory_cache: dict[tuple[str, ...], np.ndarray] = {}

    # -------------------------------------------------
    # RAW embedding (no backend interaction)
    # -------------------------------------------------
    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)

        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        elif vecs.ndim != 2:
            raise ValueError(f"Unexpected embedding shape: {vecs.shape}")

        return vecs

    # -------------------------------------------------
    # Public embed (backend-aware)
    # -------------------------------------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension()), dtype=np.float32)
        # print(f"Embedding {len(texts)} texts with model '{self.model_name}'...")

        key = tuple(texts)
        if key in self._memory_cache:
            return self._memory_cache[key]

        # 1. Load from backend
        vecs, missing_idx = self.backend.get(texts, self.model_name)

        # 2. Compute missing
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_vecs = self._embed_raw(missing_texts)

            # Persist
            self.backend.put(missing_texts, new_vecs, self.model_name)

            # Fill into vecs list
            for i, v in zip(missing_idx, new_vecs):
                vecs[i] = v

        # 3. Stack
        result = np.stack(vecs, axis=0)

        # Cache in-memory
        self._memory_cache[key] = result
        return result

    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

