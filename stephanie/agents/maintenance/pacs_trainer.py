# stephanie/agents/maintenance/pacs_trainer.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.dataloaders.casebook_to_rlvr import CaseBookToRLVRDataset
from stephanie.scoring.training.pacs_trainer import PACSTrainer


class PACSTrainerAgent(BaseAgent):
    """
    Agent wrapper for the PACSTrainer.
    
    Converts CaseBooks into RLVR datasets and trains PACS models
    on them, logging results dimension by dimension.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.trainer = PACSTrainer(cfg, memory, container, logger)
        self.casebook_name = cfg.get("casebook_name", "default_casebook")
        self.verifier = cfg.get("verifier", "default_verifier")
        self.dimensions = cfg.get("dimensions", ["alignment"])

    async def run(self, context: dict) -> dict:
        results = {}
        # Build RLVR dataset from CaseBook
        dataset_builder = CaseBookToRLVRDataset(
            memory=self.memory,
            casebook_name=self.casebook_name,
            scoring=self.memory.scores,  # integrates SICQL/EBT/LLM scorers
        )
        dataset = dataset_builder.build()

        if not dataset:
            self.logger.log(
                "NoRLVRItems", {"casebook": self.casebook_name}
            )
            context[self.output_key] = {"training_stats": {}}
            return context

        # Train on each dimension
        for dimension in self.dimensions:
            self.logger.log(
                "PACSTrainingStart",
                {"dimension": dimension, "num_samples": len(dataset)},
            )

            stats = self.trainer.train(dataset, dimension)

            if "error" in stats:
                self.logger.log(
                    "PACSTrainingError",
                    {"dimension": dimension, "error": stats["error"]},
                )
                continue

            self.logger.log(
                "PACSTrainingCompleted", {"dimension": dimension, "stats": stats}
            )

            results[dimension] = stats

        # Save results into context
        context[self.output_key] = {
            "training_stats": results,
            "casebook": self.casebook_name,
        }
        return context
