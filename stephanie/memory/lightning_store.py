# stephanie/memory/lightning_store.py
from __future__ import annotations

from typing import List

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.lightning import \
    LightningORM  # simple ORM mirroring the table
from stephanie.types.lightning import LightningResult


class LightningStore(BaseSQLAlchemyStore):
    orm_model = LightningORM
    default_order_by = "id"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "agent_lightning"

    def insert(self, data: dict) -> LightningORM:
        lr = LightningResult(**data).model_dump()
        def op(s):
            row = LightningORM(**lr)
            s.add(row); s.flush()
            if self.logger: self.logger.log("LightningInserted", {"id": row.id, "kind": lr["kind"]})
            return row
        return self._run(op)

    def recent_by_run(self, run_id: str, since_step: int = 0, limit: int = 200) -> List[LightningORM]:
        def op(s):
            q = (s.query(LightningORM)
                   .filter_by(run_id=run_id)
                   .filter(LightningORM.step_idx > int(since_step))
                   .order_by(LightningORM.step_idx.asc())
                   .limit(limit))
            return q.all()
        return self._run(op)
