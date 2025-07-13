# stephanie/memory/sharpening_store.py
# stores/sharpening_store.py
from sqlalchemy.orm import Session

from stephanie.models import PromptORM
from stephanie.models.sharpening_prediction import SharpeningPredictionORM


class SharpeningStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "sharpening"

    def insert_sharpening_prediction(self, prediction_dict: dict):
        """
        Inserts a new sharpening comparison from A/B hypothesis testing
        """
        prediction = SharpeningPredictionORM(**prediction_dict)
        self.session.add(prediction)
        self.session.commit()
        self.session.refresh(prediction)

        return prediction.id
