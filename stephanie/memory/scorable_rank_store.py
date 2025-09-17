# stephanie/memory/scorable_rank_store.py
from __future__ import annotations

from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.scorable_rank import ScorableRankORM


class ScorableRankStore(BaseSQLAlchemyStore):
    orm_model = ScorableRankORM
    default_order_by = ScorableRankORM.created_at

    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "scorable_ranks"

    def name(self) -> str:
        return self.name

    def insert(self, data: dict) -> int:
        obj = ScorableRankORM(**data)
        self.session.add(obj)
        self.session.commit()
        if self.logger:
            self.logger.log("ScorableRankInserted", obj.to_dict())
        return obj.id

    def bulk_insert(self, data_list: list[dict]) -> list[int]:
        objs = [ScorableRankORM(**d) for d in data_list]
        self.session.add_all(objs)
        self.session.commit()
        if self.logger:
            self.logger.log("ScorableRankBulkInserted", {"count": len(objs)})
        return [obj.id for obj in objs]

    def get_by_query(self, query_text: str) -> list[ScorableRankORM]:
        return (
            self.session.query(ScorableRankORM)
            .filter_by(query_text=query_text)
            .order_by(ScorableRankORM.rank_score.desc())
            .all()
        )

    def get_for_scorable(self, scorable_id: str, scorable_type: str) -> list[ScorableRankORM]:
        return (
            self.session.query(ScorableRankORM)
            .filter_by(scorable_id=scorable_id, scorable_type=scorable_type)
            .order_by(ScorableRankORM.created_at.desc())
            .all()
        )
