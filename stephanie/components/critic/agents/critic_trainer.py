# stephanie/components/critic/agents/critic_trainer_agent.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.critic.model.critic_trainer import CriticTrainer

import logging
log = logging.getLogger(__name__)

class CriticTrainerAgent(BaseAgent):
    """
    Thin wrapper around TinyCriticTrainer so training can be invoked
    inside Stephanie pipelines.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.trainer = CriticTrainer(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        run_id = context.get("pipeline_run_id")

        # Grab kept columns from DB
        metric_store = self.memory.metrics  # or however you access MetricStore
        kept = []
        if metric_store:
            try:
                kept = metric_store.get_kept_columns(run_id) or []
                log.info("CriticTrainer: retrieved %d DB-locked features for run_id=%s", len(kept), run_id)
            except Exception as e:
                log.warning("MetricStore.get_kept_columns failed: %s", e)

        # Prefer in-memory locked features if available
        if kept:
            self.trainer.cfg["lock_features_names"] = kept
            # Ensure any legacy file lock is ignored
            self.trainer.cfg.pop("lock_features", None)
            log.info("CriticTrainer: using %d DB-locked features for run_id=%s", len(kept), run_id)

        # proceed as before
        result = self.trainer.train_from_dataset()
        log.info("CriticTrainer: training complete, model_path=%s", result.model_path)
        context["critic_stats"] = {
            "critic_model_path": str(result.model_path),
            "critic_meta_path": str(result.meta_path),
            "cv": result.cv_summary,
            "holdout": result.holdout_summary,
            "model_path": result.model_path,
            "meta_path": result.meta_path,
        }
        return context
