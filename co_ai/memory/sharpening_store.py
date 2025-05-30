# stores/sharpening_store.py
from sqlalchemy.orm import Session

from co_ai.models.sharpening_prediction import SharpeningPredictionORM


class SharpeningStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "sharpening"

    def insert_sharpening_prediction(self, prediction_dict: dict, goal_text: str):
        """
        Inserts a new sharpening comparison from A/B hypothesis testing
        """
        prediction = SharpeningPredictionORM(**prediction_dict)
        goal = self.memory.goals.get_or_create({"goal_text": goal_text})
        prediction.goal_id = goal.id  # Ensure correct goal linkage

        self.session.add(prediction)
        self.session.commit()
        self.session.refresh(prediction)

        return prediction.id