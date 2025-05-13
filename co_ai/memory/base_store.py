# co_ai/memory/base_store.py
from abc import ABC, abstractmethod


class BaseStore(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def setup(self):
        """Optional: Setup logic for the store."""
        pass

    def teardown(self):
        """Optional: Cleanup logic for the store."""
        pass
