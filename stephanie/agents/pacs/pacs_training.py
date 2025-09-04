# stephanie/pipeline/stages/pacs_training.py
import os
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, PIPELINE
from stephanie.pacs import HybridSICQLAdapter, PACSConfig, PACSTrainer


class PACSTrainingAgent(BaseAgent):
    """Trains model using PACS algorithm and persists results."""

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.max_steps = cfg.get("max_steps", 1000)
        self.score_mode = cfg.get("score_mode", "critic")
        self.beta = cfg.get("beta", 1.0)
        self.group_size = cfg.get("group_size", 8)
        self.lr = cfg.get("lr", 1e-6)
        self.steps_per_reset = cfg.get("steps_per_reset", 200)
        self.base_model = cfg.get("base_model", "Qwen/Qwen2.5-1.5B")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get(GOAL, {})
        pipeline_stage = context.get(PIPELINE, "pacs_training")

        try:
            dataset = context.get("rlvr_dataset")
            verifier = context.get("verifier")
            casebook_name = context.get("casebook_name", "default_casebook")

            if not dataset or not verifier:
                self.logger.log("PACSTrainingSkipped", {
                    "stage": pipeline_stage,
                    "reason": "missing_dataset_or_verifier"
                })
                return context

            # Ensure casebook exists
            cb = self.memory.casebooks.ensure_casebook(name=casebook_name,
                                                       description=goal.get("goal_text", ""))

            # Setup base model + tokenizer
            from transformers import AutoModelForCausalLM, AutoTokenizer
            initial_actor = AutoModelForCausalLM.from_pretrained(self.base_model)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)

            # Critic (optional)
            critic = None
            if self.score_mode == "critic":
                try:
                    from stephanie.scoring.sicql_trainer import SICQLPolicyHead
                    critic = SICQLPolicyHead.load_from("models/sicql/math/")
                except Exception as e:
                    self.logger.log("CriticLoadFailed", {"error": str(e)})
                    critic = None

            # Policy adapter
            policy = HybridSICQLAdapter(
                actor_lm=initial_actor,
                tokenizer=tokenizer,
                critic_head=critic
            )

            # PACS config
            pacs_cfg = PACSConfig(
                score_mode=self.score_mode,
                beta=self.beta,
                group_size=self.group_size,
                steps_per_reset=self.steps_per_reset,
                lr=self.lr,
            )

            # Trainer
            trainer = PACSTrainer(
                policy=policy,
                cfg=pacs_cfg,
                verifier=verifier
            )

            self.logger.log("PACSTrainingStarted", {
                "stage": pipeline_stage,
                "casebook": cb.name,
                "goal": goal.get("goal_text", ""),
                "max_steps": self.max_steps,
                "score_mode": self.score_mode,
                "beta": self.beta,
                "group_size": self.group_size,
            })

            # Train
            stats = trainer.train(dataset, max_steps=self.max_steps)

            # Save trained model
            trained_model_path = os.path.join("models", "pacs", f"{cb.name}_trained")
            os.makedirs(trained_model_path, exist_ok=True)
            policy.actor_lm.save_pretrained(trained_model_path)

            # Persist report
            report_data = {
                "casebook_id": cb.id,
                "goal": goal.get("goal_text", ""),
                "score_mode": self.score_mode,
                "stats": stats,
                "trained_model_path": trained_model_path,
                "steps": self.max_steps,
            }
            try:
                self.memory.reports.add_report(report_data)
            except Exception as e:
                self.logger.log("PACSTrainingReportFailed", {"error": str(e)})

            # Update context
            context.update({
                "initial_model": initial_actor,      # pre-training snapshot
                "trained_model": policy.actor_lm,    # after PACS training
                "trained_model_path": trained_model_path,
                "pacs_trainer": trainer,
                "training_stats": stats,
            })

            self.logger.log("PACSTrainingCompleted", {
                "stage": pipeline_stage,
                "model_path": trained_model_path,
                "casebook": cb.name,
                "steps": self.max_steps,
            })

            return context

        except Exception as e:
            self.logger.log("PACSTrainingFailed", {
                "stage": pipeline_stage,
                "error": str(e)
            })
            raise
