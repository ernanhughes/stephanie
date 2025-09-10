# stephanie/models/hnsw_index.py
import os
import json
import hnswlib
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Set

_logger = logging.getLogger(__name__)

class HNSWIndex:
    """Robust wrapper for hnswlib index with metadata management and deduplication."""
    
    # Key for identifying unique entities
    ENTITY_KEY_FIELDS = ["scorable_id", "entity_text", "start", "end"]
    
    def __init__(
        self, 
        dim: int = 500, 
        index_path: str = "data/ner_retriever/index", 
        space: str = "cosine", 
        persistent: bool = True,
        max_elements: int = 100000,  # Default max elements
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
        deduplication_threshold: float = 0.98  # For similar-but-not-identical entities
    ):
        self.dim = dim
        self.index_path = index_path
        self.space = space  # "cosine", "l2", "ip"
        self.persistent = persistent
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.deduplication_threshold = deduplication_threshold
        
        # Core components
        self.index = None
        self.metadata = []  # List of metadata dicts
        self.entity_key_to_idx = {}  # For fast deduplication
        self.initiated = False
        
        # Stats tracking
        self.stats = {
            "total_adds": 0,
            "duplicates_skipped": 0,
            "updates": 0,
            "last_save_time": 0,
            "save_count": 0
        }
        
        # Load or initialize
        self._load_index()

    def _init_index(self):
        """Initialize a new index with proper parameters."""
        try:
            self.index = hnswlib.Index(space=self.space, dim=self.dim)
            self.index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.M
            )
            self.index.set_ef(self.ef_search)
            self.initiated = True
            _logger.info(
                f"Initialized new HNSW index (dim={self.dim}, space={self.space}, "
                f"max_elements={self.max_elements})"
            )
        except Exception as e:
            _logger.error(f"Failed to initialize HNSW index: {e}")
            raise

    def _get_entity_key(self, meta: Dict[str, Any]) -> str:
        """Generate a unique key for entity deduplication."""
        parts = [str(meta.get(field, "")) for field in self.ENTITY_KEY_FIELDS]
        return "|".join(parts)

    def _is_duplicate(self, meta: Dict[str, Any], embedding: np.ndarray = None) -> Tuple[bool, Optional[int]]:
        """
        Check if this entity is a duplicate of an existing one.
        Returns (is_duplicate, existing_index)
        """
        key = self._get_entity_key(meta)
        if key in self.entity_key_to_idx:
            return True, self.entity_key_to_idx[key]
        
        # Optional: check for near-duplicates using embedding similarity
        if embedding is not None and len(self.metadata) > 0:
            # This would require more complex logic with batch search
            pass
            
        return False, None

    def add(
        self, 
        embeddings: np.ndarray, 
        metadata_list: List[Dict[str, Any]], 
        save: bool = True,
        allow_updates: bool = True
    ) -> int:
        """
        Add new embeddings with metadata, with deduplication.
        
        Args:
            embeddings: Array of embeddings to add
            metadata_list: Corresponding metadata for each embedding
            save: Whether to save after adding
            allow_updates: Whether to update existing entities
            
        Returns:
            Number of new entities actually added
        """
        if embeddings.shape[0] != len(metadata_list):
            raise ValueError("Embeddings and metadata must have same length")
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        if not self.initiated:
            self._init_index()
            
        # Filter out duplicates and prepare for addition
        new_embeddings = []
        new_metadata = []
        duplicate_count = 0
        
        for i, (emb, meta) in enumerate(zip(embeddings, metadata_list)):
            is_duplicate, idx = self._is_duplicate(meta, emb)
            
            if is_duplicate:
                duplicate_count += 1
                if allow_updates and idx is not None:
                    # Update existing entry
                    self.metadata[idx] = meta
                    self.entity_key_to_idx[self._get_entity_key(meta)] = idx
                    self.stats["updates"] += 1
                continue
                
            # New entity - add to our lists
            new_embeddings.append(emb)
            new_metadata.append(meta)
            self.stats["total_adds"] += 1
            
        # Nothing new to add
        if not new_embeddings:
            if duplicate_count > 0:
                _logger.debug(f"Skipped {duplicate_count} duplicate entities")
                self.stats["duplicates_skipped"] += duplicate_count
            return 0
            
        # Convert back to numpy array
        new_embeddings = np.array(new_embeddings)
        
        # Resize index if needed
        current_count = self.index.get_current_count()
        if current_count + len(new_embeddings) > self.index.get_max_elements():
            new_max = max(
                self.max_elements,
                int((current_count + len(new_embeddings)) * 1.2)  # 20% buffer
            )
            try:
                self.index.resize_index(new_max)
                _logger.info(f"Resized HNSW index to {new_max} elements")
            except Exception as e:
                _logger.error(f"Failed to resize index: {e}")
                # Continue with current size - may hit limits later
        
        # Add to index
        start_idx = current_count
        ids = list(range(start_idx, start_idx + len(new_embeddings)))
        self.index.add_items(new_embeddings, ids)
        
        # Update metadata and tracking
        start_idx = len(self.metadata)
        for i, meta in enumerate(new_metadata):
            idx = start_idx + i
            self.metadata.append(meta)
            self.entity_key_to_idx[self._get_entity_key(meta)] = idx
            
        # Save if requested
        if save and self.persistent:
            self._save_index()
            
        _logger.info(
            f"HNSW index updated with {len(new_embeddings)} new items "
            f"({duplicate_count} duplicates skipped) "
            f"(total={len(self.metadata)})"
        )
        
        return len(new_embeddings)

    def search(self, query: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search index for nearest neighbors to query vector."""
        if not self.initiated or len(self.metadata) == 0:
            return []
            
        if query.ndim == 1:
            query = query.reshape(1, -1)
            
        if query.dtype != np.float32:
            query = query.astype(np.float32)
            
        # Perform search
        labels, distances = self.index.knn_query(query, k=k)
        
        # Process results
        results = []
        for label, dist in zip(labels[0], distances[0]):
            if label < len(self.metadata):
                meta = self.metadata[label].copy()
                meta["distance"] = float(dist)
                
                # Convert to similarity score based on space
                if self.space == "cosine":
                    meta["similarity"] = 1 - float(dist)  # similarity, higher = better
                elif self.space == "ip":
                    meta["similarity"] = float(dist)  # inner product
                else:  # l2
                    meta["similarity"] = 1.0 / (1.0 + float(dist))  # convert distance to similarity
                
                results.append(meta)
                
        return results

    def _save_index(self, force: bool = False, min_interval: int = 30) -> bool:
        """
        Save index and metadata to disk with rate limiting.
        
        Args:
            force: Force save regardless of time since last save
            min_interval: Minimum seconds between saves
            
        Returns:
            Whether a save was performed
        """
        # Rate limiting for frequent saves
        current_time = time.time()
        if not force and (current_time - self.stats["last_save_time"] < min_interval):
            return False
            
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save index
            self.index.save_index(f"{self.index_path}.bin")
            
            # Save metadata
            with open(f"{self.index_path}_metadata.json", "w") as f:
                json.dump(self.metadata, f, indent=2)
                
            # Save entity key mapping for faster reload
            with open(f"{self.index_path}_keymap.json", "w") as f:
                json.dump({k: int(v) for k, v in self.entity_key_to_idx.items()}, f)
                
            self.stats["last_save_time"] = current_time
            self.stats["save_count"] += 1
            
            _logger.info(
                f"Saved HNSW index: {self.index_path}.bin "
                f"({len(self.metadata)} items, {self.stats['save_count']} total saves)"
            )
            return True
            
        except Exception as e:
            _logger.error(f"Failed to save index: {e}")
            return False

    def _load_index(self):
        """Load index and metadata from disk with validation."""
        index_file = f"{self.index_path}.bin"
        meta_file = f"{self.index_path}_metadata.json"
        keymap_file = f"{self.index_path}_keymap.json"
        
        # Try to load existing index
        if os.path.exists(index_file) and os.path.exists(meta_file):
            try:
                # Initialize index structure
                self._init_index()
                
                # Load index data
                self.index.load_index(index_file)
                self.index.set_ef(self.ef_search)
                
                # Load metadata
                with open(meta_file, "r") as f:
                    self.metadata = json.load(f)
                
                # Load keymap if available
                if os.path.exists(keymap_file):
                    with open(keymap_file, "r") as f:
                        self.entity_key_to_idx = {k: int(v) for k, v in json.load(f).items()}
                else:
                    # Rebuild keymap from metadata
                    self.entity_key_to_idx = {}
                    for idx, meta in enumerate(self.metadata):
                        self.entity_key_to_idx[self._get_entity_key(meta)] = idx
                
                self.initiated = True
                _logger.info(
                    f"Loaded HNSW index with {len(self.metadata)} items "
                    f"(max_elements={self.index.get_max_elements()})"
                )
                return
                
            except Exception as e:
                _logger.error(f"Failed to load index: {e}")
                # Don't raise - we'll create a new index
                self._init_index()
        
        # No existing index or failed to load
        if not self.initiated:
            self._init_index()
            _logger.info("Created new HNSW index (no existing data found)")
            
            # Initialize empty metadata
            self.metadata = []
            self.entity_key_to_idx = {}

    def get_stats(self) -> Dict[str, Any]:
        """Return detailed stats for monitoring."""
        return {
            "total_entities": len(self.metadata),
            "unique_scorables": len(set(m.get("scorable_id") for m in self.metadata if m.get("scorable_id"))),
            "index_initiated": self.initiated,
            "current_count": self.index.get_current_count() if self.initiated else 0,
            "max_elements": self.index.get_max_elements() if self.initiated else 0,
            "duplicates_skipped": self.stats["duplicates_skipped"],
            "updates": self.stats["updates"],
            "total_adds": self.stats["total_adds"],
            "save_count": self.stats["save_count"],
            "last_save_time": self.stats["last_save_time"]
        }

    def optimize(self):
        """Optimize index for search performance."""
        if not self.initiated or len(self.metadata) == 0:
            return
            
        try:
            self.index.reorder()
            _logger.info("HNSW index optimized for search performance")
        except Exception as e:
            _logger.error(f"Failed to optimize index: {e}")

    def reset(self):
        """Completely reset the index (clear all data)."""
        self._init_index()
        self.metadata = []
        self.entity_key_to_idx = {}
        self.stats = {
            "total_adds": 0,
            "duplicates_skipped": 0,
            "updates": 0,
            "last_save_time": 0,
            "save_count": 0
        }
        _logger.info("HNSW index completely reset")
        
        # Clear files
        for ext in [".bin", "_metadata.json", "_keymap.json"]:
            try:
                if os.path.exists(f"{self.index_path}{ext}"):
                    os.remove(f"{self.index_path}{ext}")
            except Exception as e:
                _logger.error(f"Failed to delete {self.index_path}{ext}: {e}")

    def flush(self):
        """Force save index to disk."""
        self._save_index(force=True)

    def set_query_params(self, ef_search: int = 50):
        """Set search-time parameters."""
        self.ef_search = ef_search
        if self.index is not None:
            self.index.set_ef(ef_search)