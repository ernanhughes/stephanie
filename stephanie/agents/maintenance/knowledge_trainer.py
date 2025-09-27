# stephanie/agents/maintenance/knowledge_trainer.py
from __future__ import annotations

import time
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.dataloaders.knowledge_pair_builder import KnowledgePairBuilder
from stephanie.scoring.calibration import ScoreCalibrator
from stephanie.scoring.training.knowledge_trainer import KnowledgeTrainer


class KnowledgeTrainerAgent(BaseAgent):
    """
    Agent that trains the MR.Q DPO-lite Knowledge Scorer (reward head on frozen embeddings).
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Builder mines (pos,neg) pairs for "knowledge" from chat turns
        self.pair_builder = KnowledgePairBuilder(
            memory=memory,
            min_entity_overlap=int(
                cfg.get("knowledge", {}).get("min_entity_overlap", 1)
            ),
            seed=int(cfg.get("seed", 1337)),
        )

        self.trainer = KnowledgeTrainer(
            cfg=cfg,
            memory=memory,
            container=container,
            logger=logger,
        )

        # Defaults; overridable via cfg["knowledge"]
        kcfg = cfg.get("knowledge", {}) or {}
        self.use_pair_cache = kcfg.get("use_pair_cache", True)
        self.force_pair_refresh = kcfg.get("force_pair_refresh", False)

        self.min_star_pos = int(kcfg.get("min_star_pos", 2))
        self.max_star_neg = int(kcfg.get("max_star_neg", -1))
        self.limit_pairs = int(kcfg.get("limit_pairs", 500))
        self.max_negs_per_pos = int(kcfg.get("max_negs_per_pos", 3))
        self.shuffle_pairs = bool(kcfg.get("shuffle_pairs", True))
        self.show_progress = cfg.get("progress", {}).get("enabled", True)
        self.progress_refresh = int(
            cfg.get("progress", {}).get("refresh", 10)
        )  # refresh every N batches

    async def run(self, context: dict) -> dict:
        """
        Entry point: build contrastive pairs, train reward head, persist artifact, and return stats.
        context may include:
          - goal: {id, goal_text}   (optional; current version trains global knowledge head)
        """
        t0 = time.time()
        results: Dict[str, Any] = {}
        self.logger.log(
            "KnowledgeTrainerAgentStarted",
            {
                "min_star_pos": self.min_star_pos,
                "max_star_neg": self.max_star_neg,
                "limit_pairs": self.limit_pairs,
                "max_negs_per_pos": self.max_negs_per_pos,
                "shuffle_pairs": self.shuffle_pairs,
            },
        )

        calibrator_path = self.cfg.get(
            "calibrator_path", "config/models/calibrators/knowledge.json"
        )
        try:
            calibrator = ScoreCalibrator.load(calibrator_path)
            self.pair_builder.set_calibrator(calibrator)
            self.logger.log("CalibratorLoaded", {"path": calibrator_path})
        except Exception as e:
            self.logger.log("CalibratorLoadFailed", {"error": str(e)})
            calibrator = None

        # 1) Build pairs
        # important output A need to be preferred over B
        pairs = self.pair_builder.build_pairs(
            min_star_pos=self.min_star_pos,
            max_star_neg=self.max_star_neg,
            limit=self.limit_pairs,
            max_negs_per_pos=self.max_negs_per_pos,
            shuffle=self.shuffle_pairs,
            # force_refresh=self.force_pair_refresh
        )

        if not pairs:
            self.logger.log("KnowledgePairsEmpty", {"reason": "no pairs"})
            context["training_stats"] = {"knowledge": {"error": "no_pairs"}}
            return context

        # 2) Train + persist
        stats = self.trainer.train(pairs)
        if (
            calibrator
            and hasattr(self.trainer, "human_scores")
            and hasattr(self.trainer, "ai_scores")
        ):
            quality = calibrator.evaluate(
                self.trainer.human_scores, self.trainer.ai_scores
            )
            stats.update(
                {
                    "calibration_mse": quality["mse"],
                    "calibration_r2": quality["r2"],
                    "calibration_accuracy": quality["accuracy"],
                }
            )
        stats.update(
            {
                "trained_pairs": len(pairs),
                "beta": self.trainer.beta,
                "margin": self.trainer.margin,
                "epochs": self.trainer.epochs,
                "batch_size": self.trainer.batch_size,
                "human_pair_acc": stats.get(
                    "best_val_pair_acc_h", float("nan")
                ),
                "ai_pair_acc": stats.get("best_val_pair_acc_a", float("nan")),
                "alignment_mse": stats.get("alignment_mse", float("nan")),
                "disagreement_rate": stats.get(
                    "disagreement_rate", float("nan")
                ),
                "blend_ratio": stats.get("blend_ratio", 0.6),
            }
        )

        if "error" in stats:
            self.logger.log("KnowledgeTrainingError", stats)
            context["training_stats"] = {"knowledge": stats}
            return context
        # 3) (Optional) register artifact in memory registry
        self.memory.models.register(
            name="knowledge",
            path=stats["model_path"],
            meta={
                "avg_loss": stats.get("avg_loss"),
                "best_val_pair_acc": stats.get("best_val_pair_acc"),
                "dim": stats.get("dim"),
                "hdim": stats.get("hdim"),
                "beta": stats.get("beta"),
                "margin": stats.get("margin"),
                "trained_pairs": stats.get("trained_pairs"),
                "timestamp": stats.get("timestamp"),
                "aux_features": stats.get("aux_features"),
                "embedding_type": stats.get("embedding_type"),
                "version": stats.get("version"),
            },
        )

        # 4) Return stats
        elapsed = time.time() - t0
        stats["elapsed_sec"] = elapsed
        results["knowledge"] = stats
        context["training_stats"] = results
        return context
