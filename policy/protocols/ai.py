from typing import Protocol, Any

class ScorableAI(Protocol):
    def __call__(self, state: dict) -> Any:
        ...
