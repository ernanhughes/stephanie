# stephanie/components/risk/attr_sink_orm.py
from __future__ import annotations

from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from stephanie.models.evaluation_attribute import EvaluationAttributeORM  # keep your actual import

class ORMAttrSink:
    def __init__(self, session: Session, evaluation_id: int, prefix: str = "hall."):
        self.session = session
        self.evaluation_id = evaluation_id
        self.prefix = prefix

    def write_many(self, items: List[Tuple[str, float]]) -> None:
        now = datetime.utcnow()
        rows = []
        for k, v in items:
            if v is None or (isinstance(v, float) and (v != v)):  # allow NaNs? skip them
                continue
            rows.append(
                EvaluationAttributeORM(
                    evaluation_id=self.evaluation_id,
                    key=f"{self.prefix}{k}",
                    value=float(v),
                    created_at=now,
                )
            )
        if rows:
            self.session.add_all(rows)
            self.session.commit()
