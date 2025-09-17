# stephanie/memory/dynamic_scorable_store.py
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.dynamic_scorable import DynamicScorableORM


class DynamicScorableStore(BaseSQLAlchemyStore):
    orm_model = DynamicScorableORM
    default_order_by = DynamicScorableORM.created_at.desc()
    
    def __init__(self, session: Session, logger=None):
        super().__init__(session, logger)
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
        try:
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
            self.session.add(record)
            self.session.commit()
            return record
        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("DynamicScorableAddFailed", {"error": str(e)})
            raise

    def get_by_run(self, run_id: str) -> List[DynamicScorableORM]:
        return (
            self.session.query(DynamicScorableORM)
            .filter_by(run_id=run_id)
            .all()
        )

    def get_by_pipeline_run(
        self, pipeline_run_id: str
    ) -> List[DynamicScorableORM]:
        return (
            self.session.query(DynamicScorableORM)
            .filter_by(pipeline_run_id=pipeline_run_id)
            .all()
        )

    def get_by_case(self, case_id: int) -> List[DynamicScorableORM]:
        return (
            self.session.query(DynamicScorableORM)
            .filter_by(case_id=case_id)
            .all()
        )
    
    def children_of(self, *, source_id: int, source_type: str, limit: int = 100):
        return (
            self.session.query(DynamicScorableORM)
            .filter_by(source_scorable_id=source_id,
                       source_scorable_type=source_type)
            .order_by(DynamicScorableORM.created_at.desc())
            .limit(limit)
            .all()
        )

    def latest_child(self, *, source_id: int, source_type: str):
        return (
            self.session.query(DynamicScorableORM)
            .filter_by(source_scorable_id=source_id,
                       source_scorable_type=source_type)
            .order_by(DynamicScorableORM.created_at.desc())
            .first()
        )


    def get_latest_by_source_pointer(
        self,
        *,
        source: str,
        source_scorable_type: str,
        source_scorable_id: int,
    ) -> DynamicScorableORM | None:
        try:
            return (
                self.session.query(DynamicScorableORM)
                .filter(
                    DynamicScorableORM.source == source,
                    DynamicScorableORM.source_scorable_type == source_scorable_type,
                    DynamicScorableORM.source_scorable_id == int(source_scorable_id),
                )
                .order_by(desc(DynamicScorableORM.created_at), desc(DynamicScorableORM.id))
                .first()
            )
        except Exception as e:
            if self.logger:
                self.logger.log("DynScorableLookupError", {
                    "error": str(e),
                    "source": source,
                    "stype": source_scorable_type,
                    "sid": source_scorable_id,
                })
            return None

    def list_by_source_pointer(
        self,
        *,
        source: str,
        source_scorable_type: str,
        source_scorable_id: int,
        limit: int = 20,
    ) -> list[DynamicScorableORM]:
        try:
            return (
                self.session.query(DynamicScorableORM)
                .filter(
                    DynamicScorableORM.source == source,
                    DynamicScorableORM.source_scorable_type == source_scorable_type,
                    DynamicScorableORM.source_scorable_id == int(source_scorable_id),
                )
                .order_by(desc(DynamicScorableORM.created_at), desc(DynamicScorableORM.id))
                .limit(limit)
                .all()
            )
        except Exception as e:
            if self.logger:
                self.logger.log("DynScorableListError", {
                    "error": str(e),
                    "source": source,
                    "stype": source_scorable_type,
                    "sid": source_scorable_id,
                })
            return []
