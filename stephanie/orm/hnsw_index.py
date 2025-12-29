# stephanie/orm/hnsw_index.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hnswlib
import numpy as np

from stephanie.utils.hash_utils import hash_text

log = logging.getLogger(__name__)

class HNSWIndex:
    """Robust wrapper for hnswlib index with metadata management and deduplication."""

    ENTITY_KEY_FIELDS = ["scorable_id", "entity_text", "start", "end"]

    def __init__(
        self,
        dim: int = 500,
        index_path: str = "data/ner_retriever/index",  # prefix or directory (we normalize)
        space: str = "cosine",
        persistent: bool = True,
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
        deduplication_threshold: float = 0.98,
    ):
        self.dim = int(dim)
        self.space = str(space)
        self.persistent = bool(persistent)
        self.max_elements = int(max_elements)
        self.ef_construction = int(ef_construction)
        self.M = int(M)
        self.ef_search = int(ef_search)
        self.deduplication_threshold = float(deduplication_threshold)

        # --- Normalize the path to an absolute FILE PREFIX
        self.index_prefix = self._normalize_prefix(index_path)
        self.index_bin = f"{self.index_prefix}.bin"
        self.meta_file = f"{self.index_prefix}_metadata.json"
        self.keymap_file = f"{self.index_prefix}_keymap.json"
        Path(self.index_prefix).parent.mkdir(parents=True, exist_ok=True)
        log.debug(f"HNSWIndex prefix resolved to: {self.index_prefix}")

        # Core state
        self.index: Optional[hnswlib.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.entity_key_to_idx: Dict[str, int] = {}
        self.initiated = False
        self.node_id_to_idx: Dict[str, int] = {}
        # Stats tracking
        self.stats = {
            "total_adds": 0,
            "duplicates_skipped": 0,
            "updates": 0,
            "last_save_time": 0.0,
            "save_count": 0,
        }

        self._load_index_or_init()
        # save on exit of application
        # atexit.register(self.flush)

    @property
    def ntotal(self) -> int:
        return self.index.get_current_count() if self.index is not None else 0

    def all_metadata(self) -> List[Dict[str, Any]]:
        return list(self.metadata)


    # ---------- Path handling ----------
    def _normalize_prefix(self, p: str) -> str:
        p = Path(p).expanduser()
        if p.suffix == ".bin":
            # If user passed a file, strip suffix to use as prefix
            p = p.with_suffix("")
        if p.is_dir() or str(p).endswith(os.sep):
            # If directory, derive a stable filename inside it
            # include space & dim to avoid cross-space conflicts
            p = p / f"index_{self.space}_{self.dim}"
        return str(p.resolve())

    # ---------- Index lifecycle ----------
    def _new_index(self) -> hnswlib.Index:
        idx = hnswlib.Index(space=self.space, dim=self.dim)
        idx.init_index(max_elements=self.max_elements, ef_construction=self.ef_construction, M=self.M)
        idx.set_ef(self.ef_search)
        return idx

    def _load_index_or_init(self) -> None:
        """Load if files exist; otherwise initialize empty index."""
        if os.path.exists(self.index_bin) and os.path.exists(self.meta_file):
            try:
                self.index = hnswlib.Index(space=self.space, dim=self.dim)
                # NOTE: do NOT call init_index() before load_index()
                self.index.load_index(self.index_bin, max_elements=self.max_elements)
                self.index.set_ef(self.ef_search)
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                if os.path.exists(self.keymap_file):
                    with open(self.keymap_file, "r", encoding="utf-8") as f:
                        self.entity_key_to_idx = {k: int(v) for k, v in json.load(f).items()}
                else:
                    self._rebuild_keymap()

                # Validate label↔metadata alignment
                cur = self.index.get_current_count()
                if cur != len(self.metadata):
                    log.error(f"HNSW label/metadata mismatch: index_count={cur}, meta_len={len(self.metadata)}. "
                                  f"Refusing to proceed to avoid misaligned search results.")
                    # safest route: rebuild fresh
                    self.index = self._new_index()
                    self.metadata = []
                    self.entity_key_to_idx = {}
                    log.warning("Started a NEW empty index due to mismatch.")
                else:
                    log.debug(f"Loaded HNSW index with {len(self.metadata)} items (max={self.index.get_max_elements()})")

                self.initiated = True
                return
            except Exception as e:
                log.exception(f"Failed to load HNSW index; starting fresh. Reason: {e}")

        # Fresh index
        self.index = self._new_index()
        self.initiated = True
        self.metadata = []
        self.entity_key_to_idx = {}
        log.debug("Created new HNSW index (no existing data found)")

    
    # ---------- Keys & dedup ----------
    def _get_entity_key(self, meta: Dict[str, Any]) -> str:
        # Prefer explicit unique IDs if present
        if "node_id" in meta and meta["node_id"]:
            return f"node_id::{meta['node_id']}"
        # Try canonical entity tuple
        parts = [str(meta.get(f, "")) for f in self.ENTITY_KEY_FIELDS]
        if any(parts):
            return "|".join(parts)
        # Fallback: hash of text-like fields
        basis = (meta.get("entity_text") or meta.get("text") or "") + "::" + str(meta.get("scorable_id") or "")
        return "hash::" + hash_text(basis)[:16]

    def _is_duplicate(self, meta: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
        key = self._get_entity_key(meta)
        if key in self.entity_key_to_idx:
            return True, self.entity_key_to_idx[key]
        return False, None

    # ---------- Public API ----------
    def add(
        self,
        embeddings: np.ndarray | List[float],
        metadata_list: List[Dict[str, Any]] | Dict[str, Any],
        save: bool = True,
        allow_updates: bool = True,
    ) -> int:
        """
        Add new items. Accepts a single embedding+metadata or batches.
        Labels are always assigned so that label == index in self.metadata.
        """
        if isinstance(metadata_list, dict):
            metadata_list = [metadata_list]
        if isinstance(embeddings, list):
            embeddings = np.asarray([embeddings], dtype=np.float32)
        elif isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1).astype(np.float32)
        else:
            embeddings = embeddings.astype(np.float32)

        if embeddings.shape[0] != len(metadata_list):
            raise ValueError("Embeddings and metadata must have same length")

        if not self.initiated:
            self._load_index_or_init()

        # Guard against label/meta misalignment
        cur = self.index.get_current_count()
        if cur != len(self.metadata):
            log.error(f"Refusing to add: label/meta mismatch (index={cur}, meta={len(self.metadata)}).")
            return 0

        new_embeddings = []
        new_meta = []
        dups = 0
        for emb, meta in zip(embeddings, metadata_list):
            is_dup, idx = self._is_duplicate(meta)
            if is_dup:
                dups += 1
                if allow_updates and idx is not None:
                    self.metadata[idx] = meta
                    self.entity_key_to_idx[self._get_entity_key(meta)] = idx
                    self.stats["updates"] += 1
                continue
            new_embeddings.append(emb)
            new_meta.append(meta)

        if not new_embeddings:
            if dups:
                self.stats["duplicates_skipped"] += dups
                log.debug(f"Skipped {dups} duplicates")
            return 0

        new_embeddings = np.vstack(new_embeddings).astype(np.float32)

        # Ensure capacity
        needed = len(new_embeddings)
        if cur + needed > self.index.get_max_elements():
            new_max = int(max(self.index.get_max_elements() * 1.2, cur + needed * 2))
            try:
                self.index.resize_index(new_max)
                log.debug(f"Resized HNSW index to {new_max} elements")
            except Exception as e:
                log.error(f"Failed to resize index: {e}")

        # Assign labels to match metadata index positions
        start_label = len(self.metadata)
        ids = np.arange(start_label, start_label + needed)
        self.index.add_items(new_embeddings, ids)

        # Append metadata in the same order
        for i, meta in enumerate(new_meta):
            idx = start_label + i
            self.metadata.append(meta)
            self.entity_key_to_idx[self._get_entity_key(meta)] = idx

        self.stats["total_adds"] += needed

        if save and self.persistent:
            self._save_index()

        log.debug(
            f"HNSW index updated with {needed} new items "
            f"({dups} duplicates skipped) (total={len(self.metadata)})"
        )
        return needed

    def search(self, query: np.ndarray | List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Return metadata dicts augmented with distance/similarity."""
        if not self.initiated or len(self.metadata) == 0:
            return []
        if isinstance(query, list):
            query = np.asarray([query], dtype=np.float32)
        elif isinstance(query, np.ndarray) and query.ndim == 1:
            query = query.reshape(1, -1).astype(np.float32)
        else:
            query = query.astype(np.float32)

        k_req = int(k)
        ntotal = self.ntotal
        k_eff = max(1, min(int(k), ntotal if ntotal > 0 else k_req))

        labels, dists = self.index.knn_query(query, k=k_eff)
        out = []
        for lbl, dist in zip(labels[0], dists[0]):
            if 0 <= lbl < len(self.metadata):
                meta = dict(self.metadata[lbl])  # copy
                meta["distance"] = float(dist)
                if self.space == "cosine":
                    meta["similarity"] = 1 - float(dist)
                elif self.space == "ip":
                    meta["similarity"] = float(dist)
                else:
                    meta["similarity"] = 1.0 / (1.0 + float(dist))
                out.append(meta)
        return out
    
    def search_tuples(self, query: np.ndarray | List[float], k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Adapter: return (node_id, score, metadata) tuples for KG callers.
        - node_id: prefer metadata['node_id'], otherwise fall back to our dedup key
        - score: prefer calibrated_similarity if present, else similarity
        """
        rows = self.search(query, k=k) or []
        out: List[Tuple[str, float, Dict[str, Any]]] = []
        for meta in rows:
            nid = str(meta.get("node_id") or self._get_entity_key(meta))
            score = float(meta.get("calibrated_similarity", meta.get("similarity", 0.0)))
            out.append((nid, score, meta))
        return out


    # ---------- Save / Load ----------
    def _atomic_json_dump(self, path: str, obj: Any) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _save_index(self, force: bool = False, min_interval: int = 30) -> bool:
        if not self.persistent:
            return False
        now = time.time()

        files_exist = os.path.exists(self.index_bin) and os.path.exists(self.meta_file)
        first_save = self.stats["save_count"] == 0 or not files_exist
        must_save = force or first_save

        if not must_save and (now - self.stats["last_save_time"] < min_interval):
            return False

        try:
            Path(self.index_prefix).parent.mkdir(parents=True, exist_ok=True)
            self.index.save_index(self.index_bin)
            self._atomic_json_dump(self.meta_file, self.metadata)
            self._atomic_json_dump(self.keymap_file, {k: int(v) for k, v in self.entity_key_to_idx.items()})
            self.stats["last_save_time"] = now
            self.stats["save_count"] += 1
            log.debug(f"Saved HNSW index to {self.index_bin} ({len(self.metadata)} items, saves={self.stats['save_count']})")
            return True
        except Exception as e:
            log.exception(f"Failed to save HNSW index: {e}")
            return False

    def flush(self) -> None:
        self._save_index(force=True)

    # ---------- Maintenance / Stats ----------
    def get_stats(self) -> Dict[str, Any]:
        cur = self.index.get_current_count() if self.initiated else 0
        maxel = self.index.get_max_elements() if self.initiated else 0
        return {
            "total_entities": len(self.metadata),
            "index_initiated": self.initiated,
            "current_count": cur,
            "max_elements": maxel,
            "duplicates_skipped": self.stats["duplicates_skipped"],
            "updates": self.stats["updates"],
            "total_adds": self.stats["total_adds"],
            "save_count": self.stats["save_count"],
            "last_save_time": self.stats["last_save_time"],
            "prefix": self.index_prefix,
        }

    def optimize(self) -> None:
        if self.initiated and len(self.metadata) > 0:
            try:
                self.index.reorder()
                log.debug("HNSW index optimized")
            except Exception as e:
                log.exception(f"Failed to optimize index: {e}")

    def reset(self) -> None:
        self.index = self._new_index()
        self.metadata = []
        self.entity_key_to_idx = {}
        self.stats.update({"total_adds": 0, "duplicates_skipped": 0, "updates": 0, "last_save_time": 0.0, "save_count": 0})
        log.debug("HNSW index completely reset")
        for f in (self.index_bin, self.meta_file, self.keymap_file):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                log.error(f"Failed to delete {f}: {e}")

    def set_query_params(self, ef_search: int = 50) -> None:
        self.ef_search = int(ef_search)
        if self.index is not None:
            self.index.set_ef(self.ef_search)

    # ---------- Convenience ----------

    def _rebuild_keymap(self) -> None:
        self.entity_key_to_idx = {}
        self.node_id_to_idx = {}
        for i, meta in enumerate(self.metadata):
            key = self._get_entity_key(meta)
            self.entity_key_to_idx[key] = i
            nid = meta.get("node_id")
            if nid:
                self.node_id_to_idx[str(nid)] = i

    def has_metadata(self, key: str) -> bool:
        # direct node_id hit
        if key in self.node_id_to_idx:
            return True
        # "node_id::<id>" form
        if key.startswith("node_id::") and key in self.entity_key_to_idx:
            return True
        # slow fallback
        return any(
            m.get("node_id") == key or self._get_entity_key(m) == key
            for m in self.metadata
        )
