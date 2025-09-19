# stephanie/memory/dynamic_scorable_store.py
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import desc

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.dynamic_scorable import DynamicScorableORM


class DynamicScorableStore(BaseSQLAlchemyStore):
    orm_model = DynamicScorableORM
    default_order_by = "created_at"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "dynamic_scorables"

    def name(self) -> str:
        return self.name

    def add(
        self,
        pipeline_run_id: str,
        scorable_type: str,
        source: str,
        text: str = None,
        case_id: Optional[int] = None,
        meta: Optional[dict] = None,
        source_scorable_id: int | None = None,
        source_scorable_type: str | None = None,
    ) -> DynamicScorableORM:
        def op():
            with self._scope() as s:
                record = DynamicScorableORM(
                    pipeline_run_id=pipeline_run_id,
                    case_id=case_id,
                    scorable_type=scorable_type,
                    source=source,
                    text=text,
                    meta=meta or {},
                    source_scorable_id=source_scorable_id,
                    source_scorable_type=source_scorable_type,
                )
                s.add(record)
                s.flush()
                return record
        return self._run(op)

    def get_by_run(self, run_id: str) -> List[DynamicScorableORM]:
        return self._run(
            lambda: self._scope().query(DynamicScorableORM).filter_by(run_id=run_id).all(),
            default=[],
        )

    def get_by_pipeline_run(self, pipeline_run_id: str) -> List[DynamicScorableORM]:
        return self._run(
            lambda: self._scope().query(DynamicScorableORM)
            .filter_by(pipeline_run_id=pipeline_run_id)
            .all(),
            default=[],
        )

    def get_by_case(self, case_id: int) -> List[DynamicScorableORM]:
        return self._run(
            lambda: self._scope().query(DynamicScorableORM).filter_by(case_id=case_id).all(),
            default=[],
        )

    def children_of(self, *, source_id: int, source_type: str, limit: int = 100):
        return self._run(
            lambda: self._scope().query(DynamicScorableORM)
            .filter_by(source_scorable_id=source_id, source_scorable_type=source_type)
            .order_by(DynamicScorableORM.created_at.desc())
            .limit(limit)
            .all(),
            default=[],
        )

    def latest_child(self, *, source_id: int, source_type: str):
        return self._run(
            lambda: self._scope().query(DynamicScorableORM)
            .filter_by(source_scorable_id=source_id, source_scorable_type=source_type)
            .order_by(DynamicScorableORM.created_at.desc())
            .first(),
        )

    def get_latest_by_source_pointer(
        self,
        *,
        source: str,
        source_scorable_type: str,
        source_scorable_id: int,
    ) -> DynamicScorableORM | None:
        return self._run(
            lambda: self._scope().query(DynamicScorableORM)
            .filter(
                DynamicScorableORM.source == source,
                DynamicScorableORM.source_scorable_type == source_scorable_type,
                DynamicScorableORM.source_scorable_id == int(source_scorable_id),
            )
            .order_by(desc(DynamicScorableORM.created_at), desc(DynamicScorableORM.id))
            .first(),
        )

    def list_by_source_pointer(
        self,
        *,
        source: str,
        source_scorable_type: str,
        source_scorable_id: int,
        limit: int = 20,
    ) -> list[DynamicScorableORM]:
        return self._run(
            lambda: self._scope().query(DynamicScorableORM)
            .filter(
                DynamicScorableORM.source == source,
                DynamicScorableORM.source_scorable_type == source_scorable_type,
                DynamicScorableORM.source_scorable_id == int(source_scorable_id),
            )
            .order_by(desc(DynamicScorableORM.created_at), desc(DynamicScorableORM.id))
            .limit(limit)
            .all(),
            default=[],
        )
