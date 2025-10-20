# stephanie/utils/lru_cache.py
from __future__ import annotations
from collections import OrderedDict
from threading import RLock
from typing import Generic, TypeVar, Optional, Iterable

K = TypeVar("K")
V = TypeVar("V")

class SimpleLRUCache(Generic[K, V]):
    """
    Minimal LRU with dict-like API:
      - cache[key] = value
      - value = cache[key]              # raises KeyError if missing
      - value = cache.get(key, default)
      - key in cache
      - len(cache), cache.clear(), cache.pop(key, default)
    Thread-safe via a re-entrant lock.
    """
    def __init__(self, max_size: int = 1024):
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self._data: OrderedDict[K, V] = OrderedDict()
        self._max = int(max_size)
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._data

    def __getitem__(self, key: K) -> V:
        with self._lock:
            val = self._data[key]  # may raise KeyError
            # mark as recently used
            self._data.move_to_end(key, last=True)
            return val

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key, last=True)
            self._data[key] = value
            if len(self._data) > self._max:
                # evict least-recently-used (front)
                self._data.popitem(last=False)

    # dict-compatible helpers
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key, last=True)
                return self._data[key]
            return default

    def set(self, key: K, value: V) -> None:
        self.__setitem__(key, value)

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            if key in self._data:
                return self._data.pop(key)
            return default

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def keys(self) -> Iterable[K]:
        with self._lock:
            return list(self._data.keys())
