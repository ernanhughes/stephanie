from stephanie.agents.base_agent import BaseAgent


class Extract_infoAgent(BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        # Implement agent logic here
        return context