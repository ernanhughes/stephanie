"""
Chat Import Agent

This agent handles the import of chat conversations from various formats (JSON/HTML)
into Stephanie's memory system. It serves as the operational wrapper that coordinates
the import process within the broader pipeline workflow.

Key Responsibilities:
- Initializes import configuration and optional database purging
- Executes the chat import process via the chat_importer tool
- Provides comprehensive logging of import operations
- Updates execution context to signal successful import completion

The agent ensures chat data is properly ingested and available for downstream
processing including casebook creation and knowledge extraction.
"""

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.tools.chat_importer import import_conversations


class ChatImportAgent(BaseAgent):
    """
    Agent that imports chat conversations from a directory
    and stores them as CaseBooks in Stephanie's memory.
    """

    def __init__(self, cfg, memory, container, logger):
        # Initialize parent class with configuration, memory, container and logger
        super().__init__(cfg, memory, container, logger)
        
        # Set import path from config or use default
        self.import_path = cfg.get("import_path", "data/chats")
        
        # Optionally purge existing conversations if configured
        if cfg.get("purge_existing", False):
            self.memory.chats.purge_all(True)

    async def run(self, context: dict) -> dict:
        # Extract goal from context for logging purposes
        goal = context.get(GOAL, {})
        
        # Log the start of import operation
        self.logger.log("ChatImportStart", {
            "import_path": self.import_path,
            "goal": goal.get("goal_text") if goal else None
        })

        # Execute the import process
        try:
            # Call the import_conversations tool to handle actual import
            summary = import_conversations(self.memory, self.import_path, context=context)

            # Log detailed summary of import results
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
            # Log any errors during import process
            self.logger.log("ChatImportError", {
                "error": str(e),
                "import_path": self.import_path
            })
            raise

        # Update context to indicate successful import
        context["chat_imported"] = True
        return context