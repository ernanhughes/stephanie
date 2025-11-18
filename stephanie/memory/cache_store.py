# stephanie/memory/cache_store.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.cache_entry import CacheEntryORM


class CacheStore(BaseSQLAlchemyStore):
    """
    Portable SQLAlchemy L2 cache store.

    Defaults
    --------
    - TTL check: sliding by `accessed_at` (configurable).
    - Value stored as bytes for speed; JSON helper methods provided.
    - No dialect-specific UPSERT; uses update-then-insert pattern with race-safe fallback.
    """

    orm_model = CacheEntryORM
    default_order_by = None

    def __init__(
        self,
        session_maker,
        logger=None,
        *,
        ttl_seconds: int = 24 * 60 * 60,
        sliding_ttl: bool = True,
    ):
        super().__init__(session_maker, logger)
        self.name = "zmq_cache"
        self._ttl = int(ttl_seconds)
        self._sliding = bool(sliding_ttl)

    # ---------------------------------------------------------------------
    # Core (bytes) API
    # ---------------------------------------------------------------------
    def get_bytes(
        self,
        key: str,
        *,
        scope: Optional[str] = None,
        ttl_override: Optional[int] = None,
    ) -> Optional[bytes]:
        """
        Return bytes if present and not expired (by created/accessed per mode).
        Updates accessed_at on hit when sliding_ttl=True.
        """
        now = time.time()
        ttl_s = int(ttl_override or self._ttl)
        cutoff = now - ttl_s

        def op(s):
            q = s.query(CacheEntryORM).filter(CacheEntryORM.key == key)
            if scope is not None:
                q = q.filter(CacheEntryORM.scope == scope)

            # TTL predicate (portable)
            if self._sliding:
                q = q.filter(CacheEntryORM.accessed_at > cutoff)
            else:
                q = q.filter(CacheEntryORM.created_at > cutoff)

            row = q.first()
            if not row:
                return None

            val = row.value_bytes
            if val is None and row.value_json is not None:
                # Fallback: if stored as JSON, serialize to bytes
                val = json.dumps(row.value_json, ensure_ascii=False).encode(
                    "utf-8"
                )

            # Sliding TTL: touch last access
            if self._sliding:
                row.accessed_at = now

            return val

        return self._run(op)

    def put_bytes(
        self,
        key: str,
        value: bytes,
        *,
        scope: Optional[str] = None,
        ttl_override: Optional[int] = None,
    ) -> None:
        """
        Portable upsert: try UPDATE, if 0 rows affected then INSERT.
        Handles create-or-touch semantics for sliding TTL.
        """
        now = time.time()
        ttl_s = int(ttl_override or self._ttl)

        def op(s):
            # 1) UPDATE path
            upd = s.query(CacheEntryORM).filter(CacheEntryORM.key == key)
            if scope is not None:
                upd = upd.filter(CacheEntryORM.scope == scope)

            updated = upd.update(
                {
                    CacheEntryORM.value_bytes: value,
                    CacheEntryORM.value_json: None,
                    CacheEntryORM.ttl_seconds: ttl_s,
                    CacheEntryORM.accessed_at: now,
                },
                synchronize_session=False,
            )

            if updated:
                return

            # 2) INSERT path
            obj = CacheEntryORM(
                key=key,
                scope=scope,
                value_bytes=value,
                value_json=None,
                created_at=now,
                accessed_at=now,
                ttl_seconds=ttl_s,
            )
            try:
                s.add(obj)
                s.flush()
            except IntegrityError:
                # Race: someone inserted the same key; retry update
                s.rollback()
                upd2 = s.query(CacheEntryORM).filter(CacheEntryORM.key == key)
                if scope is not None:
                    upd2 = upd2.filter(CacheEntryORM.scope == scope)
                upd2.update(
                    {
                        CacheEntryORM.value_bytes: value,
                        CacheEntryORM.value_json: None,
                        CacheEntryORM.ttl_seconds: ttl_s,
                        CacheEntryORM.accessed_at: now,
                    },
                    synchronize_session=False,
                )

        self._run(op)

    def delete(self, key: str, *, scope: Optional[str] = None) -> int:
        def op(s):
            q = s.query(CacheEntryORM).filter(CacheEntryORM.key == key)
            if scope is not None:
                q = q.filter(CacheEntryORM.scope == scope)
            return q.delete(synchronize_session=False)

        return self._run(op)

    def count(self, *, scope: Optional[str] = None) -> int:
        def op(s):
            q = s.query(func.count(CacheEntryORM.key))
            if scope is not None:
                q = q.filter(CacheEntryORM.scope == scope)
            return int(q.scalar() or 0)

        return self._run(op)

    # ---------------------------------------------------------------------
    # JSON helpers (optional)
    # ---------------------------------------------------------------------
    def get_json(
        self,
        key: str,
        *,
        scope: Optional[str] = None,
        ttl_override: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        ttl_s = int(ttl_override or self._ttl)
        cutoff = now - ttl_s

        def op(s):
            q = s.query(CacheEntryORM).filter(CacheEntryORM.key == key)
            if scope is not None:
                q = q.filter(CacheEntryORM.scope == scope)

            if self._sliding:
                q = q.filter(CacheEntryORM.accessed_at > cutoff)
            else:
                q = q.filter(CacheEntryORM.created_at > cutoff)

            row = q.first()
            if not row:
                return None

            if self._sliding:
                row.accessed_at = now

            if row.value_json is not None:
                return row.value_json
            if row.value_bytes is not None:
                try:
                    return json.loads(row.value_bytes.decode("utf-8"))
                except Exception:
                    return None
            return None

        return self._run(op)

    def put_json(
        self,
        key: str,
        value: Dict[str, Any],
        *,
        scope: Optional[str] = None,
        ttl_override: Optional[int] = None,
    ) -> None:
        now = time.time()
        ttl_s = int(ttl_override or self._ttl)

        def op(s):
            upd = s.query(CacheEntryORM).filter(CacheEntryORM.key == key)
            if scope is not None:
                upd = upd.filter(CacheEntryORM.scope == scope)
            updated = upd.update(
                {
                    CacheEntryORM.value_json: value,
                    CacheEntryORM.value_bytes: None,
                    CacheEntryORM.ttl_seconds: ttl_s,
                    CacheEntryORM.accessed_at: now,
                },
                synchronize_session=False,
            )
            if updated:
                return
            obj = CacheEntryORM(
                key=key,
                scope=scope,
                value_bytes=None,
                value_json=value,
                created_at=now,
                accessed_at=now,
                ttl_seconds=ttl_s,
            )
            try:
                s.add(obj)
                s.flush()
            except IntegrityError:
                s.rollback()
                upd2 = s.query(CacheEntryORM).filter(CacheEntryORM.key == key)
                if scope is not None:
                    upd2 = upd2.filter(CacheEntryORM.scope == scope)
                upd2.update(
                    {
                        CacheEntryORM.value_json: value,
                        CacheEntryORM.value_bytes: None,
                        CacheEntryORM.ttl_seconds: ttl_s,
                        CacheEntryORM.accessed_at: now,
                    },
                    synchronize_session=False,
                )

        self._run(op)

    # ---------------------------------------------------------------------
    # Maintenance / retention
    # ---------------------------------------------------------------------
    def delete_older_than(
        self, cutoff_ts: float, *, scope: Optional[str] = None
    ) -> int:
        def op(s):
            q = s.query(CacheEntryORM).filter(
                CacheEntryORM.accessed_at < float(cutoff_ts)
            )
            if scope is not None:
                q = q.filter(CacheEntryORM.scope == scope)
            return q.delete(synchronize_session=False)

        return self._run(op)

    def enforce_capacity(
        self,
        max_entries: int,
        *,
        scope: Optional[str] = None,
        batch_size: int = 1000,
    ) -> int:
        """
        Portable capacity enforcement:
        - Find keys beyond capacity ordered by accessed_at ASC (coldest first)
        - Delete in batches
        """

        def op(s):
            base = s.query(CacheEntryORM.key)
            if scope is not None:
                base = base.filter(CacheEntryORM.scope == scope)
            total = (
                s.query(func.count(CacheEntryORM.key)).filter(
                    CacheEntryORM.scope == scope
                )
                if scope is not None
                else s.query(func.count(CacheEntryORM.key))
            ).scalar() or 0
            if total <= max_entries:
                return 0

            # select victims
            victims = (
                (
                    s.query(CacheEntryORM.key).filter(
                        CacheEntryORM.scope == scope
                    )
                    if scope is not None
                    else s.query(CacheEntryORM.key)
                )
                .order_by(CacheEntryORM.accessed_at.asc())
                .offset(max_entries)
                .limit(batch_size)
                .all()
            )
            keys = [k for (k,) in victims]
            if not keys:
                return 0
            return (
                s.query(CacheEntryORM)
                .filter(CacheEntryORM.key.in_(keys))
                .delete(synchronize_session=False)
            )

        return self._run(op)

    # ---------------------------------------------------------------------
    # Convenience (subject+payload → key). Keep in store to keep service slim.
    # ---------------------------------------------------------------------
    @staticmethod
    def stable_key(subject: str, payload: Any, *, version: str = "v1") -> str:
        """
        Deterministic, order-independent key for RPC caching.
        """
        import hashlib
        import json as _json

        def to_bytes(obj: Any) -> bytes:
            if isinstance(obj, (bytes, bytearray, memoryview)):
                return bytes(obj)
            if isinstance(obj, str):
                return obj.encode("utf-8")
            return _json.dumps(obj, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )

        h = hashlib.blake2s(digest_size=16)
        h.update(subject.encode("utf-8"))
        h.update(b"\x00")
        h.update(to_bytes(payload))
        h.update(b"\x00")
        h.update(version.encode("utf-8"))
        return h.hexdigest()
