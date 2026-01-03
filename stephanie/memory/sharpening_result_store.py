# stephanie/memory/sharpening_result_store.py
from __future__ import annotations

from typing import Any, Dict, List

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.sharpening_result import SharpeningResultORM


class SharpeningResultStore(BaseSQLAlchemyStore):
    orm_model = SharpeningResultORM
    default_order_by = "created_at"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "sharpening_results"

    def insert(self, payload: Dict[str, Any]) -> SharpeningResultORM:
        def op(s):
            obj = SharpeningResultORM(**payload)
            s.add(obj)
            s.flush()
            if self.logger:
                self.logger.log("SharpeningResultInserted", obj.to_dict())
            return obj
        return self._run(op)

    def list_recent(self, limit: int = 100) -> List[SharpeningResultORM]:
        def op(s):
            return s.query(SharpeningResultORM).order_by(SharpeningResultORM.created_at.desc()).limit(limit).all()
        return self._run(op)
