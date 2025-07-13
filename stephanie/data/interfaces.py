# stephanie/data/interfaces.py
from abc import ABC, abstractmethod


class DataSource(ABC):
    @abstractmethod
    def get_training_pairs(self, goal: str, limit: int) -> list[dict]:
        pass

    @abstractmethod
    def get_prompt_examples(self, goal: str, limit: int) -> list[dict]:
        pass
