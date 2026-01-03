# stephanie/tools/base_tool.py
from __future__ import annotations

from typing import Any, Dict


class BaseTask:
    """
    Standard interface for reusable tasks.
    Tasks are itmized workloads like "import paper", "build reference graph", etc.
    They often wrap Tools which do the actual logic.
    """

    name: str = "base_task"

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

    async def run(self, scorable, context: Dict[str, Any]):
        """
        Apply task logic to a Scorable.
        Must be overriden by concrete tasks.
        """
        raise NotImplementedError
