# stephanie/memory/sharpening_store.py
from __future__ import annotations

from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.sharpening_prediction import SharpeningPredictionORM


class SharpeningStore(BaseSQLAlchemyStore):
    orm_model = SharpeningPredictionORM
    default_order_by = SharpeningPredictionORM.created_at
    
    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "sharpening"

    def name(self) -> str:
        return self.name

    def insert_sharpening_prediction(self, prediction_dict: dict):
        """
        Inserts a new sharpening comparison from A/B hypothesis testing
        """
        prediction = SharpeningPredictionORM(**prediction_dict)
        self.session.add(prediction)
        self.session.commit()
        self.session.refresh(prediction)

        return prediction.id
