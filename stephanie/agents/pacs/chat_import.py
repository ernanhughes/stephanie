from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.tools.chat_importer import import_conversations

class ChatImportAgent(BaseAgent):
    """
    Agent that imports chat conversations from a directory
    and stores them as CaseBooks in Stephanie's memory.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.import_path = cfg.get("import_path", "data/chats")

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, {})
        self.logger.log("ChatImportStart", {
            "import_path": self.import_path,
            "goal": goal.get("goal_text") if goal else None
        })

        # Run the importer
        try:
            import_conversations(self.memory, self.import_path, context=context)
            self.logger.log("ChatImportSuccess", {
                "import_path": self.import_path
            })
        except Exception as e:
            self.logger.log("ChatImportError", {
                "error": str(e),
                "import_path": self.import_path
            })
            raise

        # Update context (you could pass back new casebook IDs here if desired)
        context["chat_imported"] = True
        return context
