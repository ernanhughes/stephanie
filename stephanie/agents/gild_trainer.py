# stephanie/agents/learning/gild_trainer.py

import os
import json
import torch
import torch.nn.functional as F
from datetime import datetime

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.hrm_scorer import HRMScorer
from stephanie.scoring.mrq.preference_pair_builder import PreferencePairBuilder
from stephanie.scoring.scorable_factory import ScorableFactory
from stephanie.scoring.training.sicql_trainer import SICQLTrainer
from stephanie.utils.model_locator import ModelLocator
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.ep_hrm_scorer import (
    EpistemicPlanHRMScorer,
)  # Adjust import


class GILDTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.use_gild = cfg.get(
            "use_gild", True
        )  # Should default to True if this agent runs
        self.beta = cfg.get("beta", 1.0)  # Temperature for advantage weighting
        self.learning_rate = cfg.get("learning_rate", 1e-4)
        self.epochs = cfg.get(
            "gild_epochs", 5
        )  # Number of passes over the data
        self.batch_size = cfg.get("gild_batch_size", 32)

        self.model_path = cfg.get("model_path", "models")
        self.target_type = cfg.get("target_type", "plan_trace")
        self.embedding_type = self.memory.embedding.type
        self.version = cfg.get("model_version", "v1")

        # --- Paths and Data Handling ---
        # If data was dumped to file, we need the path
        self.gild_data_file_path = cfg.get(
            "gild_data_file_path"
        )  # Fallback, ideally comes from context

        # If not provided, we can set a default path        # --- Training Components ---
        self.optimizer = None  # Will be initialized when model is loaded

        self.dimensions = cfg.get("dimensions", [])
        self.pair_builder = PreferencePairBuilder(memory.session, logger)

        self.hrm_scorer = HRMScorer(cfg.get("hrm", {}), memory, logger)
        self.sicql_scorer = SICQLScorer(
            cfg.get("sicql", {}), memory, logger
        )
        self.epistemic_plan_hrm_scorer = EpistemicPlanHRMScorer(
            cfg.get("epistemic_plan_hrm", {}), memory, logger
        )

        self.logger.log(
            "GILDTrainerAgentInitialized",
            {
                "use_gild": self.use_gild,
                "beta": self.beta,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                # Add other relevant config
            },
        )

    async def run(self, context: dict) -> dict:
        """
        Main GILD training loop using signals from PolicySynthesisAgent.
        1. Load GILD training signals (from context or file).
        2. Load the current SICQL model (specifically the pi_head).
        3. Prepare data (reconstruct states, organize advantages).
        4. Perform GILD Advantage-Weighted Regression update.
        5. Save the updated model.
        6. (Optional) Trigger re-evaluation/re-analysis.
        """
        try:
            self.logger.log("GILDTrainingStarted", {})

            # --- 1. Load GILD Signals ---
            gild_signals = self._load_gild_signals(context)
            if not gild_signals or not gild_signals.get("sicql_advantages"):
                self.logger.log(
                    "GILDTrainingWarning",
                    {"message": "No GILD signals found. Skipping training."},
                )
                context["gild_training_results"] = {
                    "status": "skipped_no_data"
                }
                return context

            sicql_advantages_data = gild_signals["sicql_advantages"]
            self.logger.log(
                "GILDTrainingDataLoaded",
                {"num_examples": len(sicql_advantages_data)},
            )

            # --- 2. Load Current SICQL Model ---
            # Determine which dimensions/models need updating
            # (Assume for now we update all dimensions present in the data)
            dimensions_to_update = list(
                set(item["dimension"] for item in sicql_advantages_data)
            )

            training_results = {}
            for dimension in dimensions_to_update:
                model = self.sicql_scorer.models.get(dimension)
                # Initialize optimizer for this model's pi_head parameters
                self.optimizer = torch.optim.Adam(
                    model.pi_head.parameters(),
                    lr=self.learning_rate,
                )

                # --- 3. Prepare Data for this Dimension ---
                dim_specific_data = [
                    item
                    for item in sicql_advantages_data
                    if item["dimension"] == dimension
                ]
                prepared_data = self._prepare_training_data(dim_specific_data)
                if not prepared_data:
                    self.logger.log(
                        "GILDTrainingDataError",
                        {
                            "dimension": dimension,
                            "error": "Failed to prepare data",
                        },
                    )
                    training_results[dimension] = {
                        "status": "failed",
                        "error": "Data preparation error",
                    }
                    continue

                # --- 4. Perform GILD Training ---
                epoch_losses = []
                model.pi_head.train()  # Set to training mode
                for epoch in range(self.epochs):
                    epoch_loss = self._run_training_epoch(model, prepared_data)
                    epoch_losses.append(epoch_loss)
                    self.logger.log(
                        "GILDTrainingEpochCompleted",
                        {
                            "dimension": dimension,
                            "epoch": epoch + 1,
                            "loss": epoch_loss,
                        },
                    )

                model.pi_head.eval()  # Set back to eval mode

                # --- 5. Save Updated Model ---
                save_success = self.sicql_manager.save_model(
                    model, dimension=dimension, model_type="sicql"
                )
                if save_success:
                    self.logger.log(
                        "GILDTrainingModelSaved", {"dimension": dimension}
                    )
                    training_results[dimension] = {
                        "status": "success",
                        "final_loss": epoch_losses[-1]
                        if epoch_losses
                        else None,
                        "loss_history": epoch_losses,
                    }
                else:
                    self.logger.log(
                        "GILDTrainingModelSaveFailed", {"dimension": dimension}
                    )
                    training_results[dimension] = {
                        "status": "completed_with_save_error",
                        "final_loss": epoch_losses[-1]
                        if epoch_losses
                        else None,
                        "loss_history": epoch_losses,
                    }

            # --- 6. Store Results in Context ---
            context["gild_training_results"] = {
                "overall_status": "completed",
                "per_dimension_results": training_results,
                "training_timestamp": datetime.now().isoformat(),
            }

            # --- 7. (Optional) Trigger Re-evaluation ---
            # This is more complex and depends on your supervisor/pipeline logic.
            # You could set a flag in the context:
            # context['trigger_re_evaluation'] = True
            # Or add specific documents/dimensions that need re-scoring based on training.
            # The supervisor would then need logic to check for this flag and run the analysis pipeline again.

            self.logger.log(
                "GILDTrainingCompleted", {"results_summary": training_results}
            )
            return context

        except Exception as e:
            self.logger.log("GILDTrainingFailed", {"error": str(e)})
            context["gild_training_results"] = {
                "status": "failed",
                "error": str(e),
            }
            # Depending on robustness needs, re-raise or return context
            raise  # Re-raise to stop pipeline on critical GILD failure?

    def _load_gild_signals(self, context: dict) -> dict:
        """Load GILD signals from context or file."""
        # 1. Try loading directly from context (if not dumped)
        signals = context.get("policy_synthesis_results", {}).get(
            "gild_signals"
        )
        if signals:
            self.logger.log("GILDDataLoadedFromContext", {})
            return signals

        # 2. Check if data was dumped and load from file
        # The PolicySynthesisAgent might have put the file path in the context
        psr = context.get("policy_synthesis_results", {})
        if (
            isinstance(psr, dict)
            and psr.get("large_data_dumped")
            and "dumped_to_file" in psr
        ):
            file_path = psr["dumped_to_file"]
        else:
            # Fallback to config path
            file_path = self.gild_data_file_path

        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    signals = json.load(f)
                self.logger.log(
                    "GILDDataLoadedFromFile", {"file_path": file_path}
                )
                return signals
            except Exception as e:
                self.logger.log(
                    "GILDDataLoadFromFileFailed",
                    {"file_path": file_path, "error": str(e)},
                )

        return {}

    def _prepare_training_data(self, sicql_advantages_data: list) -> list:
        """
        Prepare data for training: reconstruct states, organize tensors.
        This is a critical step requiring access to embeddings.
        """
        prepared_data = []
        for item in sicql_advantages_data:
            try:
                target_id = item["target_id"]
                target_type = item["target_type"]
                advantage = float(item["advantage"])  # Ensure it's a float
                dimension = item["dimension"]
                evaluation_id = item["evaluation_id"]  # Optional ID for tracking
                goal = self.memory.evaluations.get_goal(
                    evaluation_id
                )  # You need to implement this
                scorable = ScorableFactory.from_id(self.memory,
                    target_type,
                    target_id
                )  # You need to None implement this

                if not goal or not scorable:
                    self.logger.log(
                        "GILDDataPrepWarning",
                        {
                            "message": "Could not retrieve text for state reconstruction",
                            "target_id": target_id,
                            "target_type": target_type,
                        },
                    )
                    continue  # Skip this item


                with torch.no_grad(): # Usually, you get the *current* model's prediction without gradients
                    sicql_outputs = self.sicql_scorer(goal.to_dict(), scorable, dimension)
                    # sicql_outputs is the dictionary: {"q_value": ..., "state_value": ..., ...}
                state_z = self.sicql_scorer.encode(goal.to_dict(), scorable, dimension)
                prepared_data.append(
                    {
                        "q_value": sicql_outputs["q_value"].item(),
                        "state_value": sicql_outputs["state_value"].item(),  # Get the state value
                        "advantage": torch.tensor(
                            advantage, dtype=torch.float32
                        ),  # Tensor
                        "state_z": state_z, 
                        "target_id": target_id,
                        "target_type": target_type,
                        "dimension": dimension,
                        "evaluation_id": evaluation_id }
                )
            except Exception as e:
                self.logger.log(
                    "GILDDataPrepItemFailed",
                    {"target_id": item.get("target_id"), "error": str(e)},
                )
                # Continue with other items

        self.logger.log(
            "GILDDataPreparationCompleted",
            {
                "prepared_items": len(prepared_data),
                "total_input_items": len(sicql_advantages_data),
            },
        )
        return prepared_data

    def _run_training_epoch(self, model, prepared_data: list) -> float:
        """Run one epoch of GILD training."""
        total_loss = 0.0
        num_batches = 0

        # Simple batching (you might want a proper DataLoader)
        for i in range(0, len(prepared_data), self.batch_size):
            batch = prepared_data[i : i + self.batch_size]

            # Aggregate batch data
            batch_states = torch.stack(
                [item["state_z"] for item in batch]
            )  # Shape: (batch_size, z_dim)
            batch_advantages = torch.stack(
                [item["advantage"] for item in batch]
            )  # Shape: (batch_size,)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass through the policy head only
            # model.pi_head should take state_z and output action_logits
            action_logits = model.pi_head(
                batch_states
            )  # Shape: (batch_size, action_dim)

            # --- Core GILD Update ---
            # Calculate log probabilities
            log_probs = F.log_softmax(
                action_logits, dim=-1
            )  # Shape: (batch_size, action_dim)

            # Calculate weights from advantages
            # Ensure advantages are detached and have correct shape for broadcasting
            weights = torch.exp(
                self.beta * batch_advantages.detach()
            )  # Shape: (batch_size,)
            weights = weights / (
                weights.sum() + 1e-8
            )  # Normalize weights (optional but often done)
            weights = weights.unsqueeze(
                -1
            )  # Shape: (batch_size, 1) for broadcasting

            # Calculate weighted imitation loss
            # We sum over actions (dim=-1) and mean over the batch
            pi_loss = -(log_probs * weights).sum(dim=-1).mean()  # Scalar loss

            # Backward pass
            pi_loss.backward()

            # Update parameters
            self.optimizer.step()

            total_loss += pi_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
