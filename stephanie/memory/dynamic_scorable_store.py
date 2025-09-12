# stephanie/memory/dynamic_scorable_store.py
from typing import List, Optional

from sqlalchemy.orm import Session

from stephanie.models.dynamic_scorable import DynamicScorableORM


class DynamicScorableStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "dynamic_scorables"

    def add(self, pipeline_run_id: str,
            scorable_type: str, source: str,
            text: str = None, case_id: Optional[int] = None,
            meta: Optional[dict] = None) -> DynamicScorableORM:
        try:
            record = DynamicScorableORM(
                pipeline_run_id=pipeline_run_id,
                case_id=case_id,
                scorable_type=scorable_type,
                source=source,
                text=text,
                meta=meta or {}
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
        return self.session.query(DynamicScorableORM).filter_by(run_id=run_id).all()

    def get_by_pipeline_run(self, pipeline_run_id: str) -> List[DynamicScorableORM]:
        return self.session.query(DynamicScorableORM).filter_by(pipeline_run_id=pipeline_run_id).all()

    def get_by_case(self, case_id: int) -> List[DynamicScorableORM]:
        return self.session.query(DynamicScorableORM).filter_by(case_id=case_id).all()
