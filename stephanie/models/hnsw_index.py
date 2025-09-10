# stephanie/models/hnsw_index.py
import os
import json
import hnswlib
import numpy as np
import logging

_logger = logging.getLogger(__name__)

class HNSWIndex:
    """Wrapper for hnswlib index with metadata management."""

    def __init__(self, dim: int = 500, index_path: str = "data/ner_retriever/index", space: str = "cosine"):
        self.dim = dim
        self.index_path = index_path
        self.space = space  # "cosine", "l2", "ip"
        self.index = None
        self.metadata = []
        self.initiated = False

        self._load_index()

    def _init_index(self, max_elements=100000, ef_construction=200, M=16):
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(50)  # tradeoff between recall/latency
        self.initiated = True

    def add(self, embeddings: np.ndarray, metadata_list, save=True):
        """Add new embeddings with metadata."""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if not self.initiated:
            self._init_index()

        start = len(self.metadata)
        ids = list(range(start, start + len(embeddings)))
        self.index.add_items(embeddings, ids)
        self.metadata.extend(metadata_list)

        if save:
            self._save_index()

        _logger.info(
            f"HNSW index updated with {len(embeddings)} new items "
            f"(total={len(self.metadata)})"
        )

    def search(self, query: np.ndarray, k=10):
        """Search index for nearest neighbors to query vector."""
        if not self.initiated or len(self.metadata) == 0:
            return []

        if query.dtype != np.float32:
            query = query.astype(np.float32)

        labels, distances = self.index.knn_query(query.reshape(1, -1), k=k)
        results = []
        for label, dist in zip(labels[0], distances[0]):
            if label < len(self.metadata):
                meta = self.metadata[label].copy()
                meta["score"] = 1 - float(dist) if self.space == "cosine" else -float(dist)
                results.append(meta)
        return results

    def _save_index(self):
        """Save index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.index.save_index(f"{self.index_path}.bin")
        with open(f"{self.index_path}_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _load_index(self):
        """Load index and metadata from disk if available."""
        index_file = f"{self.index_path}.bin"
        meta_file = f"{self.index_path}_metadata.json"
        if os.path.exists(index_file) and os.path.exists(meta_file):
            try:
                self._init_index()
                self.index.load_index(index_file)
                with open(meta_file, "r") as f:
                    self.metadata = json.load(f)
                _logger.info(f"Loaded HNSW index with {self.index.get_current_count()} items")
            except Exception as e:
                _logger.error(f"Failed to load index: {e}")
                self._init_index()
        else:
            self._init_index()

    def get_stats(self):
        """Return stats for monitoring."""
        return {
            "total_entities": len(self.metadata),
            "unique_scorables": len(set(m.get("scorable_id") for m in self.metadata)),
            "index_initiated": self.initiated,
        }
