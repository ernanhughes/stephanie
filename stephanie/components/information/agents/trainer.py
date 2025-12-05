# stephanie/components/information/agents/trainer.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Any, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.idea_store import IdeaStore
from stephanie.models.idea import Idea

log = logging.getLogger(__name__)


class IdeaRLTrainerAgent(BaseAgent):
    """
    Periodically retrains the IdeaGenerationHead using high-reward ideas.

    - Pulls top ideas from IdeaStore (by r_final).
    - Uses IdeaCriticHead.export_rl_training_data(...) to build RL samples.
    - Delegates PPO / RL fine-tuning to rl4lms_trainer service.
    - Updates the ideator model path on IdeaGenerationHead.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(
            cfg=cfg, memory=memory, container=container, logger=logger
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Fetch high-quality ideas from the store
        min_score = float(self.cfg.get("min_r_final", 0.7))
        limit = int(self.cfg.get("sample_limit", 500))

        idea_store: IdeaStore = self.memory.ideas

        top_ideas: List[Idea] = await self._async_wrap(
            idea_store.list_top_ideas,
            min_r_final=min_score,
            limit=limit,
        )

        if not top_ideas:
            log.warning(
                "IdeaRLTrainerAgent: no high-reward ideas found for RL training "
                f"(min_r_final={min_score}, limit={limit})"
            )
            context["rl_training_result"] = {
                "status": "no_data",
                "samples_used": 0,
            }
            return context

        # 2. Export RL samples using the critic head
        critic = self.container.build_agent("idea_critic_head")

        samples = await critic.export_rl_training_data(
            top_ideas,
            min_final_score=min_score,
        )

        if not samples:
            log.warning(
                "IdeaRLTrainerAgent: no valid RL samples generated "
                f"from {len(top_ideas)} top ideas"
            )
            context["rl_training_result"] = {
                "status": "no_samples",
                "samples_used": 0,
            }
            return context

        # 3. Trigger RL training via rl4lms_trainer
        rl_trainer = self.container.get("rl4lms_trainer")  # Your RL trainer service

        training_config = {
            "base_model": self.cfg.get(
                "base_model",
                "llama-3-70b-ideator-v0",
            ),
            "output_dir": f"models/ideator-rl-{int(time.time())}",
            "num_epochs": int(self.cfg.get("num_epochs", 2)),
            "batch_size": int(self.cfg.get("batch_size", 8)),
            "kl_penalty": float(self.cfg.get("kl_penalty", 0.2)),
        }

        log.info(
            "IdeaRLTrainerAgent: starting RL training "
            f"with {len(samples)} samples "
            f"(base_model={training_config['base_model']})"
        )

        new_model_path = await rl_trainer.train(
            prompts=[s["prompt"] for s in samples],
            responses=[s["response"] for s in samples],
            rewards=[float(s["reward"]) for s in samples],
            config=training_config,
        )

        # 4. Update IdeaGenerationHead to use the new model
        generator = self.container.get("idea_generation_head")
        # This method should update any internal model name / config
        await generator.update_generator_model(new_model_path)

        avg_reward = sum(float(s["reward"]) for s in samples) / len(samples)

        log.info(
            "IdeaRLTrainerAgent: RL training complete. "
            f"new_model={new_model_path}, samples_used={len(samples)}, "
            f"avg_reward={avg_reward:.3f}"
        )

        context["rl_training_result"] = {
            "status": "ok",
            "new_model": new_model_path,
            "samples_used": len(samples),
            "avg_reward": avg_reward,
        }
        return context

    async def _async_wrap(self, func, *args, **kwargs):
        """
        Run sync store methods in a threadpool to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
