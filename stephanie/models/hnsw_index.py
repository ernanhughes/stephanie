# stephanie/models/hnsw_index.py
import os
import json
import hnswlib
import numpy as np
import logging

_logger = logging.getLogger(__name__)

class HNSWIndex:
    """Wrapper for hnswlib index with metadata management."""

    def __init__(self, dim: int = 500, index_path: str = "data/ner_retriever/index", space: str = "cosine", persistent: bool = True):
        self.dim = dim
        self.index_path = index_path
        self.space = space  # "cosine", "l2", "ip"
        self.index = None
        self.metadata = []
        self.initiated = False
        self.persistent = persistent

        self._load_index()

    def _init_index(self, max_elements=100000, ef_construction=200, M=16):
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(50)  # tradeoff between recall/latency
        self.initiated = True

    def add(self, embeddings: np.ndarray, metadata_list, save=True):
        """Add new embeddings with metadata."""
        if self.index.get_max_elements() < len(self.metadata) + len(embeddings):
            self.index.resize_index(len(self.metadata) + len(embeddings) + 10000)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if not self.initiated:
            self._init_index()

        start = len(self.metadata)
        ids = list(range(start, start + len(embeddings)))
        self.index.add_items(embeddings, ids)
        self.metadata.extend(metadata_list)

        if save and self.persistent:
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
                if self.space == "cosine":
                    meta["score"] = 1 - float(dist)   # similarity, higher = better
                else:
                    meta["score"] = -float(dist)
                results.append(meta)
        return results

    def _save_index(self):
        """Save index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        _logger.info(f"Saving HNSW index: {self.index_path}.bin ({len(self.metadata)} items)")
        self.index.save_index(f"{self.index_path}.bin")
        with open(f"{self.index_path}_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _load_index(self):
        index_file = f"{self.index_path}.bin"
        meta_file = f"{self.index_path}_metadata.json"
        if os.path.exists(index_file) and os.path.exists(meta_file):
            try:
                self.index = hnswlib.Index(space=self.space, dim=self.dim)
                self.index.load_index(index_file)
                self.index.set_ef(50)
                with open(meta_file, "r") as f:
                    self.metadata = json.load(f)
                self.initiated = True
                _logger.info(f"Loaded HNSW index with {self.index.get_current_count()} items")
                return
            except Exception as e:
                _logger.error(f"Failed to load index: {e}")
                raise   # don’t auto-init here
        else:
            _logger.warning(f"No existing index found at {index_file}. You must call reset() to init.")

    def get_stats(self):
        """Return stats for monitoring."""
        return {
            "total_entities": len(self.metadata),
            "unique_scorables": len(set(m.get("scorable_id") for m in self.metadata)),
            "index_initiated": self.initiated,
        }

    def flush(self):
        self._save_index()

    def set_query_params(self, ef_search: int = 50):
        if self.index is not None:
            self.index.set_ef(ef_search)
