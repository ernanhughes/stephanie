from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class SSPActor(ABC):
    @abstractmethod
    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def update(self, feedback: Dict[str, Any]) -> None:
        pass
