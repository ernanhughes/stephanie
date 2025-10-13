# stephanie/agents/maintenance/tiny_recursion_trainer_agent.py
"""
TinyRecursionTrainerAgent
-------------------------
Trains the TinyRecursionModel using evaluation_export_view data.
Uses the same design conventions as other maintenance trainers (MRQ, SVM, etc.).
"""

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.training.tiny_recursion_trainer import TinyRecursionTrainer
from stephanie.dataloaders.tiny_recursion_data_loader import TinyRecursionDataLoader


class TinyRecursionTrainerAgent(BaseAgent):
    """
    Agent that wraps TinyRecursionTrainer.
    Fetches data from evaluation_export_view using a SQLAlchemy session.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.limit = cfg.get("limit", 5000)
        self.trainer = TinyRecursionTrainer(cfg, memory=memory, container=container, logger=logger)

    async def run(self, context: dict) -> dict:
        """
        Entry point for the agent (Arena-compatible).
        Loads reasoning samples, trains TinyRecursionModel, and records stats.
        """
        # Create a temporary session for view queries
        data_loader = TinyRecursionDataLoader(self.memory, self.logger)
        samples = data_loader.fetch_samples(limit=self.limit)

        if not samples:
            self.logger.log("TinyRecursionNoSamples", {"limit": self.limit})
            context["tiny_recursion_training"] = {"error": "no_samples_found"}
            return context

        # Train model using the collected recursion samples
        stats = self.trainer.train(samples, dimension="reasoning")

        # Save training statistics into context for Arena tracking
        context["tiny_recursion_training"] = stats
        self.logger.log("TinyRecursionTrainingComplete", {"stats": stats})
        return context
