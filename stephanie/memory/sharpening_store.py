# stephanie/memory/sharpening_store.py
from __future__ import annotations

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.sharpening_prediction import SharpeningPredictionORM


class SharpeningStore(BaseSQLAlchemyStore):
    orm_model = SharpeningPredictionORM
    default_order_by = SharpeningPredictionORM.created_at

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "sharpening"

    def name(self) -> str:
        return self.name

    def insert_sharpening_prediction(self, prediction_dict: dict) -> int:
        """
        Insert a new sharpening prediction (from A/B hypothesis testing).
        Returns the inserted row ID.
        """

        def op(s):
            prediction = SharpeningPredictionORM(**prediction_dict)
            s.add(prediction)
            s.flush()
            s.refresh(prediction)
            return prediction.id

        return self._run(op)
