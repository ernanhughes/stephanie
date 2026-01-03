# stephanie/memory/scorable_feature_store.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.scorable_feature import ScorableFeatureORM


class ScorableFeatureStore(BaseSQLAlchemyStore):
    orm_model = ScorableFeatureORM
    default_order_by = "id"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "scorable_features"

    def upsert(self, row: Dict) -> ScorableFeatureORM:
        stype, sid = row["scorable_type"], row["scorable_id"]
        def op(s):
            obj = (s.query(ScorableFeatureORM)
                     .filter_by(scorable_type=stype, scorable_id=sid)
                     .first())
            if obj:
                for k, v in row.items():
                    setattr(obj, k, v)
                action = "ScorableFeatureUpdated"
            else:
                obj = ScorableFeatureORM(**row)
                s.add(obj)
                action = "ScorableFeatureInserted"
            if self.logger:
                self.logger.log(action, {"type": stype, "id": sid, "processor": row.get("processor_version")})
            return obj
        return self._run(op)

    def upsert_many(self, rows: Iterable[Dict]) -> List[ScorableFeatureORM]:
        def op(s):
            out = []
            for row in rows:
                stype, sid = row["scorable_type"], row["scorable_id"]
                obj = (s.query(ScorableFeatureORM)
                         .filter_by(scorable_type=stype, scorable_id=sid)
                         .first())
                if obj:
                    for k, v in row.items():
                        setattr(obj, k, v)
                else:
                    obj = ScorableFeatureORM(**row)
                    s.add(obj)
                out.append(obj)
            return out
        return self._run(op)

    def get(self, scorable_type: str, scorable_id: str) -> Optional[ScorableFeatureORM]:
        def op(s):
            return (s.query(ScorableFeatureORM)
                      .filter_by(scorable_type=scorable_type, scorable_id=scorable_id)
                      .first())
        return self._run(op)

    def get_children(self, parent_type: str, parent_id: str) -> List[ScorableFeatureORM]:
        def op(s):
            q = (s.query(ScorableFeatureORM)
                   .filter_by(parent_scorable_type=parent_type, parent_scorable_id=parent_id)
                   .order_by(ScorableFeatureORM.order_in_parent.nullsLast(),
                             ScorableFeatureORM.scorable_type,
                             ScorableFeatureORM.scorable_id))
            return q.all()
        return self._run(op)

    def get_siblings(self, scorable_type: str, scorable_id: str) -> List[ScorableFeatureORM]:
        def op(s):
            me = (s.query(ScorableFeatureORM)
                    .filter_by(scorable_type=scorable_type, scorable_id=scorable_id)
                    .first())
            if not me or not me.parent_scorable_id or not me.parent_scorable_type:
                return []
            q = (s.query(ScorableFeatureORM)
                   .filter(ScorableFeatureORM.parent_scorable_type == me.parent_scorable_type,
                           ScorableFeatureORM.parent_scorable_id   == me.parent_scorable_id,
                           ~((ScorableFeatureORM.scorable_type == scorable_type) &
                             (ScorableFeatureORM.scorable_id   == scorable_id)))
                   .order_by(ScorableFeatureORM.order_in_parent.nullsLast(),
                             ScorableFeatureORM.scorable_type,
                             ScorableFeatureORM.scorable_id))
            return q.all()
        return self._run(op)

    def get_by_scorable_id(self, scorable_id: str) -> Optional[ScorableFeatureORM]:
        def op(s):
            return s.query(ScorableFeatureORM).filter_by(scorable_id=scorable_id).first()
        return self._run(op)

    # Helpful readers
    def list_by_conversation(self, conversation_id: int, limit: int = 500, offset: int = 0) -> List[ScorableFeatureORM]:
        def op(s):
            q = (s.query(ScorableFeatureORM)
                   .filter_by(conversation_id=conversation_id)
                   .order_by(ScorableFeatureORM.order_index.asc().nullsLast()))
            return q.offset(offset).limit(limit).all()
        return self._run(op)

    def list_recent(self, limit: int = 100) -> List[ScorableFeatureORM]:
        def op(s):
            q = (s.query(ScorableFeatureORM)
                   .order_by(ScorableFeatureORM.created_utc.desc().nullsLast(),
                             ScorableFeatureORM.id.desc()))
            return q.limit(limit).all()
        return self._run(op)
