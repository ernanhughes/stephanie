# stephanie/memory/base.py
from abc import ABC, abstractmethod


class BaseStore(ABC):
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger

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
