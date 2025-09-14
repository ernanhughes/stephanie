from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.tools.chat_importer import import_conversations


class ChatImportAgent(BaseAgent):
    """
    Agent that imports chat conversations from a directory
    and stores them as CaseBooks in Stephanie's memory.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.import_path = cfg.get("import_path", "data/chats")
        if cfg.get("purge_existing", True):
            self.memory.chats.purge_all(True)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, {})
        self.logger.log("ChatImportStart", {
            "import_path": self.import_path,
            "goal": goal.get("goal_text") if goal else None
        })

        # Run the importer
        try:
            summary = import_conversations(self.memory, self.import_path, context=context)

            # Log detailed summary
            self.logger.log("ChatImportSuccess", {
                "import_path": self.import_path,
                "files_processed": summary.get("files_processed", 0),
                "files_skipped": summary.get("files_skipped", 0),
                "conversations_imported": summary.get("conversations_imported", 0),
                # Optional: if you add case/scorable counts to summary, log them too
                "cases_created": summary.get("cases_created", 0),
                "scorables_created": summary.get("scorables_created", 0),
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
