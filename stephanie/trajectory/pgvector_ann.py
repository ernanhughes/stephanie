# stephanie/trajectory/pgvector_ann.py
from __future__ import annotations

from typing import List, Tuple, Union
import numpy as np
import re

TURN_TAG = "TURN"

class PgVectorANN:
    """
    Adapter over your BaseEmbeddingStore.
    Uses:
      - store.get_or_create(text) to embed queries (if query text)
      - store.find_neighbors(vector, k) to do ANN
    We identify chat turns by a tag embedded in the stored text:  [TURN:<id>].
    """

    def __init__(self, memory, turn_tag: str = TURN_TAG, top_multiplier: int = 5):
        """
        embedding_store: an instance of BaseEmbeddingStore (or compatible)
        turn_tag:        the tag used at indexing, e.g. "TURN"
        top_multiplier:  fetch extra neighbors then filter by tag
        """
        self.memory = memory
        self.turn_tag = turn_tag
        self._tag_re = re.compile(rf"\[\s*{re.escape(self.turn_tag)}\s*:(\d+)\s*\]")
        self.top_multiplier = top_multiplier

    def _ensure_vec(self, q: Union[str, np.ndarray]) -> list:
        if isinstance(q, np.ndarray):
            v = q.astype(float)
            v = v / (np.linalg.norm(v) + 1e-12)
            return v.tolist()
        # Treat as text: use your default embedder + normalize
        v = np.asarray(self.store.get_or_create(str(q)), dtype=float)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v.tolist()

    def search(self, q: Union[str, np.ndarray], k: int = 20) -> List[Tuple[int, float]]:
        """
        Returns: list of (turn_id, similarity) in descending similarity.
        Internally pulls more than k, because many rows in embeddings may be non-turn rows.
        """
        vec = self._ensure_vec(q)
        # Pull more and filter by tag so we reliably get k turn hits without schema changes.
        rows = self.memory.embedding.find_neighbors(vec, k=max(k * self.top_multiplier, k))
        out: List[Tuple[int, float]] = []
        for row in rows or []:
            txt = (row.get("text") or "")
            m = self._tag_re.search(txt)
            if not m:
                continue
            turn_id = int(m.group(1))
            score = float(row.get("score", 0.0))
            out.append((turn_id, score))
            if len(out) >= k:
                break
        return out
