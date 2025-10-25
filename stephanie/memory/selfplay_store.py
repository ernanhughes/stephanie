# stephanie/memory/selfplay_store.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.selfplay_item import SelfPlayItemORM


class SelfPlayStore(BaseSQLAlchemyStore):
    """
    Minimal append/read/purge store for self-play traces.

    Scope examples:
      - "casebook:123"
      - "paper:abc"
      - "arena:run_2025_09_01"

    Buffer names are logical streams inside a scope:
      - "pool", "beam", "winners", "metrics", etc.
    """

    orm_model = SelfPlayItemORM
    default_order_by = SelfPlayItemORM.ts_ms.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "selfplay"

    # --- writes ---

    def insert(self, item_dict: Dict[str, Any]) -> SelfPlayItemORM:
        """
        Insert a single self-play item.
        Expected keys: scope(str), buffer_name(str), ts_ms(int), data(dict)
        created_at is auto-filled by ORM default.
        """
        def op(s):
            row = SelfPlayItemORM(**item_dict)
            s.add(row)
            s.flush()
            if self.logger:
                self.logger.log("SelfPlayItemInserted", {
                    "id": row.id,
                    "scope": row.scope,
                    "buffer_name": row.buffer_name,
                    "ts_ms": int(row.ts_ms),
                })
            return row
        return self._run(op)

    def insert_many(self, items: List[Dict[str, Any]]) -> int:
        """
        Bulk insert. Returns number of rows inserted.
        """
        if not items:
            return 0

        def op(s):
            rows = [SelfPlayItemORM(**it) for it in items]
            s.bulk_save_objects(rows)
            if self.logger:
                self.logger.log("SelfPlayItemsInserted", {"count": len(rows)})
            return len(rows)
        return self._run(op)

    # --- reads ---

    def list_by_scope_buffer(
        self,
        scope: str,
        buffer_name: str,
        *,
        limit: Optional[int] = None,
        min_ts_ms: Optional[int] = None,
        order_desc: bool = True,
    ) -> List[SelfPlayItemORM]:
        """
        Return items for a given (scope, buffer_name).
        """
        def op(s):
            q = (
                s.query(SelfPlayItemORM)
                 .filter(
                     SelfPlayItemORM.scope == scope,
                     SelfPlayItemORM.buffer_name == buffer_name,
                 )
            )
            if min_ts_ms is not None:
                q = q.filter(SelfPlayItemORM.ts_ms >= int(min_ts_ms))
            q = q.order_by(
                SelfPlayItemORM.ts_ms.desc() if order_desc else SelfPlayItemORM.ts_ms.asc()
            )
            if limit is not None:
                q = q.limit(int(limit))
            return q.all()
        return self._run(op)

    def list_recent_winners(
        self,
        scope: str,
        *,
        limit: int = 10,
        min_ts_ms: Optional[int] = None,
        order_desc: bool = True,
    ) -> List[SelfPlayItemORM]:
        """
        Convenience: winners buffer.
        """
        return self.list_by_scope_buffer(
            scope=scope,
            buffer_name="winners",
            limit=limit,
            min_ts_ms=min_ts_ms,
            order_desc=order_desc,
        )

    # --- purges ---

    def purge_scope(self, scope: str) -> int:
        """
        Delete all rows for a scope.
        Returns rows deleted.
        """
        def op(s):
            res = (
                s.query(SelfPlayItemORM)
                 .filter(SelfPlayItemORM.scope == scope)
                 .delete(synchronize_session=False)
            )
            if self.logger:
                self.logger.log("SelfPlayScopePurged", {"scope": scope, "rows": int(res or 0)})
            return int(res or 0)
        return self._run(op)

    def purge_buffer(self, scope: str, buffer_name: str) -> int:
        """
        Delete all rows for a specific buffer in a scope.
        """
        def op(s):
            res = (
                s.query(SelfPlayItemORM)
                 .filter(
                     SelfPlayItemORM.scope == scope,
                     SelfPlayItemORM.buffer_name == buffer_name,
                 )
                 .delete(synchronize_session=False)
            )
            if self.logger:
                self.logger.log("SelfPlayBufferPurged", {
                    "scope": scope, "buffer_name": buffer_name, "rows": int(res or 0)
                })
            return int(res or 0)
        return self._run(op)

    def purge_older_than(self, ts_ms: int) -> int:
        """
        Delete any rows older than ts_ms across the entire table.
        """
        def op(s):
            res = (
                s.query(SelfPlayItemORM)
                 .filter(SelfPlayItemORM.ts_ms < int(ts_ms))
                 .delete(synchronize_session=False)
            )
            if self.logger:
                self.logger.log("SelfPlayPurgedOlderThan", {"ts_ms": int(ts_ms), "rows": int(res or 0)})
            return int(res or 0)
        return self._run(op)

    def purge_all(self) -> int:
        """
        Delete all rows (portable DELETE; if you prefer TRUNCATE, add a raw exec here).
        """
        def op(s): 
            res = s.query(SelfPlayItemORM).delete(synchronize_session=False)
            if self.logger:
                self.logger.log("SelfPlayPurgedAll", {"rows": int(res or 0)})
            return int(res or 0)
        return self._run(op)
