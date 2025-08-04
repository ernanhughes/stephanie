# stephanie/agents/maintenance/epistemic_plan_hrm_trainer_agent.py

import json
import os
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.plan_trace import PlanTrace
from stephanie.scoring.model.epistemic_trace_encoder import \
    EpistemicTraceEncoder
from stephanie.scoring.model.hrm_model import HRMModel
from stephanie.scoring.model_locator_mixin import ModelLocatorMixin
from stephanie.utils.trace_utils import (get_trace_score_stats,
                                         load_plan_traces_from_export_dir)


class EpistemicPlanHRMTrainerAgent(ModelLocatorMixin, BaseAgent):
    """
    Agent to train the Hierarchical Reasoning Model (HRM) specifically for evaluating
    the epistemic quality of reasoning plan traces (PlanTrace objects).

    This model takes an encoded representation of a PlanTrace and predicts a single
    score representing the overall quality of the reasoning process.
    """

    def __init__(
        self, cfg: Dict[str, Any], memory: Any = None, logger: Any = None
    ):
        super().__init__(cfg, memory, logger)
        self.model_type = "epistemic_hrm"
        self.model_path = cfg.get("model_path", "models")
        self.evaluator = "hrm"
        self.target_type = cfg.get("target_type", "plan_trace")
        self.version = cfg.get("model_version", "v1")

        # --- Configuration specific to Epistemic Plan HRM ---
        self.dim = self.memory.embedding.dim
        self.hrm_cfg = cfg.get("hrm", {})
        self.encoder_cfg= cfg.get("encoder", {})
        self.encoder_cfg["embedding_dim"] = self.dim  # For goal + final output

        
        self.dimensions = cfg.get("dimensions", [])
        self.dim = self.memory.embedding.dim
        self.export_dir = cfg.get(
            "export_dir", "reports/epistemic_plan_hrm_trainer"
        )
        self.get_trace_score_stats = get_trace_score_stats

        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # --- Instantiate the HRM Model ---
        try:
            self.hrm_model = HRMModel(
                self.hrm_cfg, logger=self.logger
            ).to(self.device)
            self.logger.log(
                "EpistemicPlanHRMModelInitialized",
                {
                    "dimensions": self.dimensions,
                    "model_config": self.hrm_cfg,
                    "device": str(self.device),
                    "model_parameters": sum(
                        p.numel() for p in self.hrm_model.parameters()
                    ),
                },
            )
        except Exception as e:
            self.logger.log(
                "EpistemicPlanHRMModelInitError",
                {
                    "message": "Failed to initialize HRMModel.",
                    "error": str(e),
                },
            )
            self.hrm_model = None
            return

        # --- Initialize Optimizer ---
        try:
            # Use AdamW as recommended by HRM paper
            self.optimizer = torch.optim.AdamW(
                self.hrm_model.parameters(), lr=self.hrm_cfg["lr"]
            )
            self.logger.log(
                "EpistemicPlanHRMOptimizerInitialized",
                {
                    "optimizer": "AdamW",
                    "learning_rate": self.hrm_cfg["lr"],
                },
            )
        except Exception as e:
            self.logger.log(
                "EpistemicPlanHRMOptimizerInitError",
                {
                    "message": "Failed to initialize optimizer.",
                    "error": str(e),
                },
            )

        # --- Loss Function ---
        self.criterion = (
            nn.MSELoss()
        )  # For regression of quality score (0.0 to 1.0)
        self.logger.log(
            "EpistemicPlanHRMLossInitialized", {"loss_function": "MSELoss"}
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log(
            "EpistemicPlanHRMTrainingStarted",
            {
                "dimensions": self.dimensions,
                "epochs": self.hrm_cfg["epochs"],
                "batch_size": self.hrm_cfg["batch_size"],
            },
        )

        # --- 1. Load and Prepare Training Data
        raw_traces_data = context.get("plan_traces", [])
        if not raw_traces_data:
            # If no traces are provided, try loading from export directory
            self.logger.log(
                "EpistemicPlanHRMTrainingNoTraces",
                {
                    "message": "No plan traces found in context['plan_traces']. Attempting to load from export directory.",
                    "export_dir": self.export_dir,
                }, 
            ) 
            raw_traces_data = load_plan_traces_from_export_dir(self.export_dir)

        if not raw_traces_data:
            error_msg = (
                "No plan traces found in context['plan_traces']. Cannot train."
            )
            self.logger.log(
                "EpistemicPlanHRMTrainingError", {"message": error_msg}
            )
            context[self.output_key] = {
                "status": "failed",
                "message": error_msg
            }   
            return context

        # Filter traces with valid targets
        # training_traces = [t for t in raw_traces_data if t.has_target_quality()]
        training_traces = raw_traces_data

        self.logger.log(
            "EpistemicPlanHRMTrainingDataPrepared",
            {
                "total_traces_received": len(raw_traces_data),
                "valid_traces_for_training": len(training_traces),
                "dimensions": self.dimensions,
            },
        )

        if not training_traces:
            error_msg = "No plan traces with valid 'target_epistemic_quality' found. Cannot train."
            self.logger.log(
                "EpistemicPlanHRMTrainingError", {"message": error_msg}
            )
            context[self.output_key] = {
                "status": "failed",
                "message": error_msg
            }   
            return context

        # --- 2. Encode Traces and Prepare Tensors ---
        try:
            # This method needs to be implemented to use EpistemicTraceEncoder
            # It should return lists of tensors: [z_trace_tensor, ...], [target_score, ...]
            encoded_inputs, target_scores = (
                self._encode_traces_and_extract_targets(training_traces)
            )

            if (
                not encoded_inputs
                or not target_scores
                or len(encoded_inputs) != len(target_scores)
            ):
                raise ValueError(
                    "Encoding process returned invalid or mismatched data."
                )

            # Convert to tensors and DataLoader
            inputs_tensor = torch.stack(encoded_inputs).to(
                self.device
            )  # Shape: (N, input_dim)
            targets_tensor = torch.tensor(
                target_scores, dtype=torch.float32
            ).to(self.device)  # Shape: (N,)
            if self.hrm_cfg["output_dim"] == 1:
                targets_tensor = targets_tensor.unsqueeze(
                    1
                )  # Shape: (N, 1) for MSE with output_dim=1

            dataset = TensorDataset(inputs_tensor, targets_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.hrm_cfg["batch_size"],
                shuffle=True,
            )

            self.logger.log(
                "EpistemicPlanHRMDataLoaderCreated",
                {
                    "num_samples": len(dataset),
                    "num_batches": len(dataloader),
                    "batch_size": self.hrm_cfg["batch_size"],
                },
            )

        except Exception as e:
            error_msg = f"Error during trace encoding or data preparation: {e}"
            self.logger.log(
                "EpistemicPlanHRMTrainingDataError",
                {
                    "message": error_msg,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            context[self.output_key] = {
                "status": "failed",
                "message": error_msg
            }   
            return context

        # --- 3. Training Loop ---
        try:
            self.hrm_model.train()  # Set model to training mode
            num_epochs = self.hrm_cfg["epochs"]

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0

                for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    # The HRMModel.forward returns (y_hat, intermediate_states)
                    y_pred, _ = self.hrm_model(
                        x_batch
                    )  # y_pred shape: (B, output_dim=1)

                    # Compute loss
                    loss = self.criterion(y_pred, y_batch)

                    # Backward pass
                    # PyTorch's autograd handles the one-step gradient approximation
                    # for the nested loop structure internally.
                    loss.backward()

                    # Update parameters
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 10 == 0:
                        self.logger.log(
                            "EpistemicPlanHRMTrainingBatch",
                            {
                                "epoch": epoch,
                                "batch": batch_idx,
                                "loss": loss.item(),
                            },
                        )

                # Log average epoch loss
                avg_epoch_loss = (
                    epoch_loss / num_batches if num_batches > 0 else 0.0
                )
                self.logger.log(
                    "EpistemicPlanHRMTrainingEpoch",
                    {
                        "epoch": epoch,
                        "avg_loss": avg_epoch_loss,
                    },
                )

            # Set model back to evaluation mode
            self.hrm_model.eval()

        except Exception as e:
            error_msg = f"Error during HRM model training loop: {e}"
            self.logger.log(
                "EpistemicPlanHRMTrainingLoopError",
                {
                    "message": error_msg,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            context[self.output_key] = {
                "status": "failed",
                "message": error_msg
            }   
            return context

        # --- 4. Save Model ---
        try:
            self._save_model()
            self.logger.log(
                "EpistemicPlanHRMTrainingCompleted",
                {
                    "final_avg_loss": round(avg_epoch_loss, 6),
                },
            )

            context[self.output_key] = {
                "status": "trained",
                "final_loss": round(avg_epoch_loss, 6),
                "message": "Epistemic Plan HRM trained successfully.",
                "epochs_trained": num_epochs,
                "samples_used": len(dataset),
            }
            return context

        except Exception as e:
            error_msg = f"Error saving trained HRM model: {e}"
            self.logger.log(
                "EpistemicPlanHRMTrainingSaveError",
                {
                    "message": error_msg,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            context[self.output_key] = {
                "status": "trained_partial",  # Model trained, but save failed
                "final_loss": round(avg_epoch_loss, 6),
                "message": error_msg,
                "epochs_trained": num_epochs,
                "samples_used": len(dataset),
            }
            return context

    def _encode_traces_and_extract_targets(
        self, traces: list[PlanTrace]
    ) -> Tuple[List[torch.Tensor], List[float]]:
        self.trace_encoder = EpistemicTraceEncoder(
            self.encoder_cfg
        ).to(self.device)

        encoded_inputs = []
        target_scores = []

        for trace in traces:
            try:
                z = self.trace_encoder(
                    trace=trace,
                    embedding_lookup_fn=self.memory.embedding.get_or_create,
                    score_stats_fn=self.get_trace_score_stats, 
                    dimensions=self.dimensions,
                )
                encoded_inputs.append(z.detach())
                target_scores.append(trace.get_target_quality())
            except Exception as e:
                self.logger.log(
                    "TraceEncodingError",
                    {
                        "trace_id": getattr(trace, "trace_id", "unknown"),
                        "error": str(e),
                    },
                )
                continue

        return encoded_inputs, target_scores

    def _save_model(self):
        """Saves the trained HRM model components using the Locator."""
        from stephanie.utils.file_utils import \
            save_json  # Assuming this utility exists

        for dimension in self.dimensions:
            locator = self.get_locator(
                dimension
            )  # From BaseAgent/ModelLocatorMixin

            # Save model state dict with a specific suffix for this trainer type
            model_save_path = locator.model_file(suffix="_hrm_epistemic.pt")
            torch.save(self.hrm_model.state_dict(), model_save_path)

            # Save configuration metadata
            meta = {
                "model_type": self.model_type,
                "dimension": dimension,
                "trainer_agent": self.__class__.__name__,
                "training_completed_at": __import__("datetime")
                    .datetime.utcnow()
                    .isoformat()
                    + "Z",

                # Explicit model architecture config
                "input_dim": self.hrm_cfg["input_dim"],
                "h_dim": self.hrm_cfg["h_dim"],
                "l_dim": self.hrm_cfg["l_dim"],
                "output_dim": self.hrm_cfg["output_dim"],
                "n_cycles": self.hrm_cfg["n_cycles"],
                "t_steps": self.hrm_cfg["t_steps"],

                # Training-specific metadata
                "lr": self.hrm_cfg["lr"],
                "epochs": self.hrm_cfg["epochs"],
                "batch_size": self.hrm_cfg["batch_size"]
            }
            meta_save_path = locator.meta_file()
            # Ensure directory exists
            os.makedirs(os.path.dirname(meta_save_path), exist_ok=True)
            save_json(meta, meta_save_path)

            self.logger.log(
                "EpistemicPlanHRMModelSaved",
                {
                    "model_path": model_save_path,
                    "meta_path": meta_save_path,
                    "dimension": dimension,
                },
            )
