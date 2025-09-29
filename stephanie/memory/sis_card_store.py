# stephanie/memory/sis_card_store.py
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.sis_card import SisCardORM


class SisCardStore(BaseSQLAlchemyStore):
    orm_model = SisCardORM
    default_order_by = SisCardORM.ts.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "sis_cards"

    def _digest(self, payload: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def upsert_payload(self, payload: Dict[str, Any]) -> SisCardORM:
        """
        payload: {scope, key, title, cards, meta?, ts?}
        """
        ts = float(payload.get("ts") or time.time())
        scope = str(payload.get("scope") or "misc")
        key = str(payload.get("key") or "unknown")
        title = payload.get("title")
        cards = payload.get("cards") or []
        meta  = payload.get("meta") or {}
        h = self._digest({"scope": scope, "key": key, "title": title, "cards": cards, "meta": meta})

        def op(s):
            existing = s.query(SisCardORM).filter_by(scope=scope, key=key, hash=h).first()
            if existing:
                return existing
            obj = SisCardORM(ts=ts, scope=scope, key=key, title=title, cards=cards, meta=meta, hash=h)
            s.add(obj); s.flush()
            if self.logger:
                self.logger.info("SisCardInserted", extra={"id": obj.id, "scope": scope, "key": key})
            return obj

        return self._run(op)

    # Queries
    def recent(self, scope: Optional[str] = None, limit: int = 100) -> List[SisCardORM]:
        def op(s):
            q = s.query(SisCardORM)
            if scope: q = q.filter(SisCardORM.scope == scope)
            return q.order_by(SisCardORM.ts.desc(), SisCardORM.id.desc()).limit(limit).all()
        return self._run(op)

    def by_key(self, key: str, limit: int = 50) -> List[SisCardORM]:
        def op(s):
            return (s.query(SisCardORM)
                      .filter(SisCardORM.key == key)
                      .order_by(SisCardORM.ts.desc(), SisCardORM.id.desc())
                      .limit(limit).all())
        return self._run(op)
