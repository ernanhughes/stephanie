from stephanie.agents.base_agent import BaseAgent
from stephanie.worldview.db.locator import WorldviewDBLocator


class WorldViewCeate(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.base_directory = cfg.get("base_directory", "worldviews")
        self.locater = WorldviewDBLocator(self.base_directory)

    async def run(self, context: dict) -> dict:
        # Implement agent logic here
        goal_text = context.get("goal", {}).get("goal_text", "")
        if not goal_text:
            self.logger.log("WorldviewCreateNoGoal", {"context": context})
            return context
        path = self.locater.create_worldview(goal_text, self.memory.session)
        context["worldview_path"] = path
        return context