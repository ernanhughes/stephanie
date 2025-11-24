# stephanie/agents/maintenance/tiny_recursion_trainer_agent.py
"""
TinyRecursionTrainerAgent
-------------------------
Trains one TinyRecursionModel **per dimension** using evaluation_export_view data.
Follows the same conventions as other maintenance trainers (MRQ, SVM, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.dataloaders.tiny_recursion_data_loader import \
    TinyRecursionDataLoader
from stephanie.scoring.training.tiny_recursion_trainer import TinyTrainer


class TinyRecursionTrainerAgent(BaseAgent):
    """
    Arena-compatible maintenance agent that wraps TinyRecursionTrainer
    and trains **one model per requested dimension**.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.limit: int = int(cfg.get("limit", 50_000))
        # default to your 5 chat-analysis dimensions
        self.dimensions: List[str] = cfg.get(
            "dimensions",
            ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"],
        )
        self.use_calibrated: bool = bool(cfg.get("use_calibrated_score", True))
        self.min_score = cfg.get("min_score", None)  # e.g., 10 to drop very-low labels

        self.trainer = TinyTrainer(cfg, memory=memory, container=container, logger=logger)
        self.data_loader = TinyRecursionDataLoader(
            memory=memory,
            logger=logger,
            use_calibrated_score=self.use_calibrated,
            min_score=self.min_score,
            show_progress=cfg.get("show_progress", True),
            label_hist_bucket=cfg.get("label_hist_bucket", 10),
            drop_missing_reflection=cfg.get("drop_missing_reflection", False),
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        For each dimension:
          1) fetch samples for that dimension,
          2) train tiny recursion model,
          3) persist model + meta via trainer,
          4) collect stats for Arena/telemetry.
        """
        results: Dict[str, Any] = {}
        total_samples = 0

        for dim in self.dimensions:
            # 1) Fetch per-dimension samples
            samples = self.data_loader.fetch_samples_for_dimension(dim, limit=self.limit)
            total_samples += len(samples)

            if not samples:
                self.logger.log("TinyRecursionNoSamplesForDimension", {"dimension": dim, "limit": self.limit})
                results[dim] = {"error": "no_samples_found", "count": 0}
                continue

            # 2) Train & save per-dimension model (trainer handles saving to models/tiny/.../{dim}/{version})
            stats = self.trainer.train(samples, dimension=dim)

            # 3) Log + collect
            self.logger.log("TinyRecursionTrainingComplete", {"dimension": dim, "stats": stats})
            results[dim] = {"count": len(samples), **stats}

        # Summary for Arena
        context["tiny_recursion_training"] = {
            "dimensions": self.dimensions,
            "total_samples": total_samples,
            "per_dimension": results,
            "model_version": getattr(self.trainer, "version", "v1"),
        }
        return context
