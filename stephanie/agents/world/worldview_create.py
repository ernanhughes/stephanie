# stephanie/agents/world/worldview_create.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.world.db.locator import WorldviewDBLocator


class WorldViewCeate(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.base_directory = cfg.get("base_directory", "worldviews")
        self.pipelines = cfg.get("pipelines", [])
        self.locater = WorldviewDBLocator(self.base_directory)

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text", "")
        path = self.locater.create_worldview(goal_text, self.memory.session)
        context["worldview_path"] = path
        context["pipelines"] = self.pipelines
        return context
