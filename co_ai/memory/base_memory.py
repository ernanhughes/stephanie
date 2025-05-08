# co_ai/memory/base_memory.py
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    @abstractmethod
    def store_hypothesis(self, hypothesis):
        """
        Stores a Hypothesis object into the system.
        Could involve logging, vector embedding, and persistence.
        """
        pass

    @abstractmethod
    def search_related(self, query: str, top_k: int = 5):
        """
        Searches memory (e.g., using pgvector or Haystack) for related hypotheses or context.
        """
        pass
