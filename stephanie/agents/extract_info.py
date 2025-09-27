# stephanie/agents/extract_info.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent


class Extract_infoAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        # Implement agent logic here
        return context