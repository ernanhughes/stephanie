# stephanie/tools/base_tool.py
from __future__ import annotations

class BaseTool:
    """
    Standard interface for reusable long-running tools.
    Tools do the actual logic; Features orchestrate them.
    """

    name: str = "base_tool"

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

    async def apply(self, scorable, context: dict):
        """
        Apply tool logic to a Scorable.
        Must be overriden by concrete tools.
        """
        raise NotImplementedError
