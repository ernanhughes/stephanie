# stephanie/services/strategy_profile_service.py
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from stephanie.services.service_protocol import Service

# ---------- StrategyProfile model ----------

DEFAULT_PACS_WEIGHTS = {"skeptic": 0.34, "editor": 0.33, "risk": 0.33}

@dataclass
class StrategyProfile:
    verification_threshold: float = 0.90
    pacs_weights: Optional[Dict[str, float]] = None
    strategy_version: int = 1
    last_updated: float = 0.0

    def __post_init__(self):
        if self.pacs_weights is None:
            self.pacs_weights = dict(DEFAULT_PACS_WEIGHTS)
        if not self.last_updated:
            self.last_updated = time.time()

    def update(
        self,
        *,
        pacs_weights: Optional[Dict[str, float]] = None,
        verification_threshold: Optional[float] = None,
    ):
        if pacs_weights is not None:
            self.pacs_weights = pacs_weights
        if verification_threshold is not None:
            self.verification_threshold = verification_threshold
        self.strategy_version += 1
        self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyProfile":
        return cls(
            verification_threshold=float(d.get("verification_threshold", 0.90)),
            pacs_weights=dict(d.get("pacs_weights", DEFAULT_PACS_WEIGHTS)),
            strategy_version=int(d.get("strategy_version", 1)),
            last_updated=float(d.get("last_updated", time.time())),
        )

# ---------- Backends ----------

class _Backend:
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
    def save(self, key: str, data: Dict[str, Any]) -> None:
        raise NotImplementedError
    def delete(self, key: str) -> None:
        raise NotImplementedError
    def health(self) -> Dict[str, Any]:
        return {}

class _FileBackend(_Backend):
    """JSON file mapping {key: profile_dict}. Atomic replace on write."""
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _read_all(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_all(self, obj: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._read_all().get(key)

    def save(self, key: str, data: Dict[str, Any]) -> None:
        with self._lock:
            all_data = self._read_all()
            all_data[key] = data
            self._write_all(all_data)

    def delete(self, key: str) -> None:
        with self._lock:
            all_data = self._read_all()
            if key in all_data:
                del all_data[key]
                self._write_all(all_data)

    def health(self) -> Dict[str, Any]:
        try:
            st = os.stat(self.path)
            return {"backend": "file", "path": self.path, "size_bytes": st.st_size}
        except Exception:
            return {"backend": "file", "path": self.path, "size_bytes": None}

class _DictBackend(_Backend):
    """In-memory mapping; optionally bound to an external dict."""
    def __init__(self, backing: Optional[Dict[str, Any]] = None, bucket_key: str = "_strategy_profiles"):
        self._store = backing if backing is not None else {}
        self._bucket_key = bucket_key
        self._store.setdefault(self._bucket_key, {})
        self._lock = threading.Lock()

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._store.get(self._bucket_key, {})).get(key)

    def save(self, key: str, data: Dict[str, Any]) -> None:
        with self._lock:
            bucket = self._store.setdefault(self._bucket_key, {})
            bucket[key] = data

    def delete(self, key: str) -> None:
        with self._lock:
            bucket = self._store.get(self._bucket_key, {})
            if key in bucket:
                del bucket[key]

    def health(self) -> Dict[str, Any]:
        with self._lock:
            bucket = self._store.get(self._bucket_key, {})
            return {"backend": "dict", "entries": len(bucket)}

# ---------- Service ----------

class StrategyProfileService(Service):
    """
    Stores and retrieves agent strategy profiles (thresholds, weights, versions).

    Why this service?
    - Agents evolve their behavior (verification thresholds, PACS weights) across runs.
    - We don’t want agents to poke at MemoryTool internals like `memory.meta`.
    - This service gives a stable API and a robust backing store (file by default).

    Backends:
      * File-backed JSON (default, atomic writes)
      * Dict-backed (only if a real dict is explicitly supplied)
    """

    def __init__(
        self,
        *,
        cfg: Optional[Dict[str, Any]],
        memory,
        logger,
        path: Optional[str] = None,
        namespace: str = "global",
        dict_backing: Optional[Dict[str, Any]] = None,  # explicit dict if you want in-process backing
    ):
        self.cfg = cfg
        self.logger = logger
        self.memory = memory
        self.namespace = namespace
        self._initialized = False

        # Backend selection (robust: do NOT touch memory.meta unless it's a real dict you passed in)
        self._mode = "file"
        if dict_backing is not None and isinstance(dict_backing, dict):
            self._backend: _Backend = _DictBackend(backing=dict_backing)
            self._mode = "dict"
        else:
            # Try to use a dict-like meta if – and only if – we can obtain it safely.
            meta_dict: Optional[Dict[str, Any]] = None
            try:
                # Access in a try/except to avoid MemoryTool raising on unknown attributes.
                candidate = getattr(memory, "meta")  # no default; catch AttributeError explicitly
                if isinstance(candidate, dict):
                    meta_dict = candidate
            except AttributeError:
                meta_dict = None
            except Exception:
                meta_dict = None

            if isinstance(meta_dict, dict):
                self._backend = _DictBackend(backing=meta_dict)
                self._mode = "memory.meta"
            else:
                # Default to file-backed store
                default_path = path or os.path.join(".cache", "strategy_profiles.json")
                self._backend = _FileBackend(default_path)
                self._mode = "file"

    # --- Service protocol ---
    def initialize(self, **kwargs) -> None:
        self._initialized = True
        try:
            if self.logger:
                self.logger.log("StrategyProfileServiceInit", {"backend": self._mode})
        except Exception:
            pass

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {},
            "dependencies": self._backend.health(),
        }

    def shutdown(self) -> None:
        self._initialized = False

    @property
    def name(self) -> str:
        return "strategy-profile-service-v1"

    # --- Public API ---
    def key_for(self, *, agent_name: str, scope: Optional[str] = None) -> str:
        sc = (scope or self.namespace).strip() or "global"
        return f"{sc}::{agent_name}"

    def load(
        self,
        *,
        agent_name: str,
        scope: Optional[str] = None,
        default: Optional[StrategyProfile] = None,
    ) -> StrategyProfile:
        key = self.key_for(agent_name=agent_name, scope=scope)
        raw = self._backend.load(key)
        if raw is None:
            return default or StrategyProfile()
        try:
            return StrategyProfile.from_dict(raw)
        except Exception:
            return StrategyProfile()

    def save(self, *, agent_name: str, profile: StrategyProfile, scope: Optional[str] = None) -> None:
        key = self.key_for(agent_name=agent_name, scope=scope)
        self._backend.save(key, profile.to_dict())

    def update(
        self,
        *,
        agent_name: str,
        scope: Optional[str] = None,
        pacs_weights: Optional[Dict[str, float]] = None,
        verification_threshold: Optional[float] = None,
    ) -> StrategyProfile:
        prof = self.load(agent_name=agent_name, scope=scope)
        prof.update(pacs_weights=pacs_weights, verification_threshold=verification_threshold)
        self.save(agent_name=agent_name, profile=prof, scope=scope)
        return prof

    # Optional admin helper
    def delete(self, *, agent_name: str, scope: Optional[str] = None) -> None:
        key = self.key_for(agent_name=agent_name, scope=scope)
        try:
            self._backend.delete(key)
        except Exception:
            pass
