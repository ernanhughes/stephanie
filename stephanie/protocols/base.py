from abc import ABC
from typing import Any, Callable, Dict, List, Optional


class Protocol(ABC):
    def run(self, input_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method that each protocol must implement.
        """
        pass

ProtocolRecord = Dict[str, Any]