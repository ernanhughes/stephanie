# stephanie/agents/maintenance/epistemic_plan_hrm_trainer_agent.py

import os
import json
import traceback
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
from stephanie.agents.base_agent import BaseAgent
from stephanie.data.plan_trace import PlanTrace
from stephanie.scoring.model.hrm_model import HRMModel
from stephanie.scoring.model.epistemic_trace_encoder import EpistemicTraceEncoder
from stephanie.scoring.model_locator_mixin import ModelLocatorMixin

# Assuming PlanTrace is available or will be passed as dict

class EpistemicPlanHRMTrainerAgent(ModelLocatorMixin, BaseAgent):
    """
    Agent to train the Hierarchical Reasoning Model (HRM) specifically for evaluating
    the epistemic quality of reasoning plan traces (PlanTrace objects).

    This model takes an encoded representation of a PlanTrace and predicts a single
    score representing the overall quality of the reasoning process.
    """
    def __init__(self, cfg: Dict[str, Any], memory: Any = None, logger: Any = None):
        super().__init__(cfg, memory, logger)
        self.model_type = "hrm"
        self.model_path = cfg.get("model_path", "models")
        self.evaluator = "hrm"
        self.target_type = cfg.get("target_type", "plan_trace")
        self.version = cfg.get("model_version", "v1")


        # --- Configuration specific to Epistemic Plan HRM ---
        self.epistemic_hrm_config = cfg.get("hrm", {})
        self.dimensions = cfg.get("dimensions", [])
        self.dim = self.memory.embedding.dim
        self.export_dir = cfg.get("export_dir", "reports/epistemic_plan_hrm_trainer")
        
        # --- Configure HRM Model ---
        # Use the specific config for this HRM instance
        # Ensure input_dim matches the output of EpistemicTraceEncoder
        self.hrm_model_config = {
            "hrm.input_dim": 256, # Must match encoder output
            "hrm.h_dim": self.epistemic_hrm_config.get("hrm.h_dim", 256),
            "hrm.l_dim": self.epistemic_hrm_config.get("hrm.l_dim", 128),
            "hrm.output_dim": self.epistemic_hrm_config.get("hrm.output_dim", 1), # Predicting 1 score
            "hrm.n_cycles": self.epistemic_hrm_config.get("hrm.n_cycles", 4),
            "hrm.t_steps": self.epistemic_hrm_config.get("hrm.t_steps", 4),
            # Training specific
            "hrm.lr": self.epistemic_hrm_config.get("hrm.lr", 1e-4),
            "hrm.epochs": self.epistemic_hrm_config.get("hrm.epochs", 20),
            "hrm.batch_size": self.epistemic_hrm_config.get("hrm.batch_size", 1),
            # Add other training params as needed
        }
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Instantiate the HRM Model ---
        try:
            self.hrm_model = HRMModel(self.hrm_model_config, logger=self.logger).to(self.device)
            self.logger.log("EpistemicPlanHRMModelInitialized", {
                "dimensions": self.dimensions,
                "model_config": {k: v for k, v in self.hrm_model_config.items() if k.startswith("hrm.")},
                "device": str(self.device),
                "model_parameters": sum(p.numel() for p in self.hrm_model.parameters())
            })
        except Exception as e:
            self.logger.log("EpistemicPlanHRMModelInitError", {
                "message": "Failed to initialize HRMModel.",
                "error": str(e),
            })
            self.hrm_model = None
            return

        # --- Initialize Optimizer ---
        try:
            # Use AdamW as recommended by HRM paper
            self.optimizer = torch.optim.AdamW(
                self.hrm_model.parameters(), 
                lr=self.hrm_model_config["hrm.lr"]
            )
            self.logger.log("EpistemicPlanHRMOptimizerInitialized", {
                "optimizer": "AdamW",
                "learning_rate": self.hrm_model_config["hrm.lr"]
            })
        except Exception as e:
             self.logger.log("EpistemicPlanHRMOptimizerInitError", {
                 "message": "Failed to initialize optimizer.",
                 "error": str(e),
             })

        # --- Loss Function ---
        self.criterion = nn.MSELoss() # For regression of quality score (0.0 to 1.0)
        self.logger.log("EpistemicPlanHRMLossInitialized", {"loss_function": "MSELoss"})

        # --- Placeholder for Trace Encoder ---
        # The actual encoding logic needs the EpistemicTraceEncoder.
        # This agent assumes it's handled elsewhere or passed in a specific way.
        # self.trace_encoder = None # Will be initialized or accessed as needed

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:

        self.logger.log("EpistemicPlanHRMTrainingStarted", { 
            "dimensions": self.dimensions,
            "epochs": self.hrm_model_config["hrm.epochs"],
            "batch_size": self.hrm_model_config["hrm.batch_size"]
        })

        # --- 1. Load and Prepare Training Da OK 
        raw_traces_data = context.get("plan_traces", [])
        if not raw_traces_data:
            # If no traces are provided, try loading from export directory
            raw_traces_data = self.load_plan_traces_from_export_dir()
        
        if not raw_traces_data:
            error_msg = "No plan traces found in context['plan_traces']. Cannot train."
            self.logger.log("EpistemicPlanHRMTrainingError", { "message": error_msg })
            return { "status": "failed", "message": error_msg, "dimensions": self.dimensions }

        # Filter traces with valid targets
        training_traces = [
            t for t in raw_traces_data 
            if (hasattr(t, 'target_epistemic_quality') and t.target_epistemic_quality is not None) or
               (isinstance(t, dict) and t.get('target_epistemic_quality') is not None)
        ]
        
        self.logger.log("EpistemicPlanHRMTrainingDataPrepared", {
            "total_traces_received": len(raw_traces_data),
            "valid_traces_for_training": len(training_traces),
            "dimensions": self.dimensions
        })

        if not training_traces:
            error_msg = "No plan traces with valid 'target_epistemic_quality' found. Cannot train."
            self.logger.log("EpistemicPlanHRMTrainingError", { "message": error_msg })
            return { "status": "failed", "message": error_msg, "dimensions": self.dimensions }

        # --- 2. Encode Traces and Prepare Tensors ---
        try:
            # This method needs to be implemented to use EpistemicTraceEncoder
            # It should return lists of tensors: [z_trace_tensor, ...], [target_score, ...]
            encoded_inputs, target_scores = self._encode_traces_and_extract_targets(training_traces)
            
            if not encoded_inputs or not target_scores or len(encoded_inputs) != len(target_scores):
                 raise ValueError("Encoding process returned invalid or mismatched data.")
            
            # Convert to tensors and DataLoader
            inputs_tensor = torch.stack(encoded_inputs).to(self.device) # Shape: (N, input_dim)
            targets_tensor = torch.tensor(target_scores, dtype=torch.float32).to(self.device) # Shape: (N,)
            if self.hrm_model_config["hrm.output_dim"] == 1:
                 targets_tensor = targets_tensor.unsqueeze(1) # Shape: (N, 1) for MSE with output_dim=1
            
            dataset = TensorDataset(inputs_tensor, targets_tensor)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.hrm_model_config["hrm.batch_size"], 
                shuffle=True
            )
            
            self.logger.log("EpistemicPlanHRMDataLoaderCreated", {
                "num_samples": len(dataset),
                "num_batches": len(dataloader),
                "batch_size": self.hrm_model_config["hrm.batch_size"],
                "dimensions": self.dimensions
            })
            
        except Exception as e:
            error_msg = f"Error during trace encoding or data preparation: {e}"
            self.logger.log("EpistemicPlanHRMTrainingDataError", {
                "message": error_msg,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "dimensions": self.dimensions
            })
            return { "status": "failed", "message": error_msg, "dimensions": self.dimensions }

        # --- 3. Training Loop ---
        try:
            self.hrm_model.train() # Set model to training mode
            num_epochs = self.hrm_model_config["hrm.epochs"]
            
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
                    y_pred, _ = self.hrm_model(x_batch) # y_pred shape: (B, output_dim=1)

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
                    
                    # if batch_idx % 10 == 0:
                    #     self.logger.log("EpistemicPlanHRMTrainingBatch", {
                    #         "epoch": epoch, "batch": batch_idx, "loss": loss.item(),
                    #         "hrm_dimension": self.hrm_dimension
                    #     })

                # Log average epoch loss
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                self.logger.log("EpistemicPlanHRMTrainingEpoch", {
                    "epoch": epoch, 
                    "avg_loss": avg_epoch_loss,
                    "dimensions": self.dimensions
                })

            # Set model back to evaluation mode
            self.hrm_model.eval()

        except Exception as e:
            error_msg = f"Error during HRM model training loop: {e}"
            self.logger.log("EpistemicPlanHRMTrainingLoopError", {
                "message": error_msg,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "dimensions": self.dimensions
            })
            return { "status": "failed", "message": error_msg, "dimensions": self.dimensions }

        # --- 4. Save Model ---
        try:
            self._save_model()
            self.logger.log("EpistemicPlanHRMTrainingCompleted", {
                "final_avg_loss": round(avg_epoch_loss, 6),
                "dimensions":  self.dimensions
            })
            
            return {
                "status": "trained",
                "final_loss": round(avg_epoch_loss, 6),
                "message": "Epistemic Plan HRM trained successfully.",
                "dimensions": self.dimensions,
                "epochs_trained": num_epochs,
                "samples_used": len(dataset)
            }

        except Exception as e:
            error_msg = f"Error saving trained HRM model: {e}"
            self.logger.log("EpistemicPlanHRMTrainingSaveError", {
                "message": error_msg,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "dimensions": self.dimensions
            })
            return {
                "status": "trained_partial", # Model trained, but save failed
                "final_loss": round(avg_epoch_loss, 6),
                "message": error_msg,
                "epochs_trained": num_epochs,
                "samples_used": len(dataset)
            }



    def _encode_traces_and_extract_targets(self, traces) -> Tuple[List[torch.Tensor], List[float]]:
        encoder_config = self.epistemic_hrm_config.get("trace_encoder", {})
        embedding_dim = self.memory.embedding.dim # it cant be anythign else
        step_hidden_dim = encoder_config.get("step_hidden_dim", 64)
        stats_input_dim = encoder_config.get("stats_input_dim", 12)
        stats_hidden_dim = encoder_config.get("stats_hidden_dim", 128)
        final_dim = encoder_config.get("final_dim", 256)

        # Initialize the hybrid encoder
        self.trace_encoder = EpistemicTraceEncoder(
            embedding_dim=embedding_dim,
            step_hidden_dim=step_hidden_dim,
            stats_input_dim=stats_input_dim,
            stats_hidden_dim=stats_hidden_dim,
            final_dim=final_dim
        ).to(self.device)

        encoded_inputs = []
        target_scores = []

        for trace in traces:
            try:
                z = self.trace_encoder(
                    trace=trace,
                    embedding_lookup_fn=self.memory.embedding.get_or_create,
                    score_stats_fn=self.get_trace_score_stats  # You’ll define this separately
                )
                encoded_inputs.append(z.detach())
                print(f"Encoded trace shape: {z.shape}")

                score = trace.target_epistemic_quality if hasattr(trace, "target_epistemic_quality") else trace.get("target_epistemic_quality")
                target_scores.append(float(score))
            except Exception as e:
                self.logger.log("TraceEncodingError", {
                    "trace_id": getattr(trace, "trace_id", "unknown"),
                    "error": str(e)
                })
                continue

        return encoded_inputs, target_scores

    def _save_model(self): 
        """Saves the trained HRM model components using the Locator."""
        from stephanie.utils.file_utils import save_json # Assuming this utility exists
        for dimension in self.dimensions:
            locator = self.get_locator(dimension) # From BaseAgent/ModelLocatorMixin

            # Save model state dict with a specific suffix for this trainer type
            model_save_path = locator.model_file(suffix="_hrm_epistemic.pt")
            torch.save(self.hrm_model.state_dict(), model_save_path)
        
            # Save configuration metadata
            meta = {
                "model_type": self.model_type,
                "dimension": dimension,
                "trainer_agent": self.__class__.__name__,
                "training_completed_at": __import__('datetime').datetime.utcnow().isoformat() + 'Z',
                # Include all relevant HRM hyperparameters
                **{k: v for k, v in self.hrm_model_config.items() if k.startswith("hrm.")},
            }
            meta_save_path = locator.meta_file()
            # Ensure directory exists
            os.makedirs(os.path.dirname(meta_save_path), exist_ok=True)
            save_json(meta, meta_save_path)
            
            self.logger.log("EpistemicPlanHRMModelSaved", {
                "model_path": model_save_path,
                "meta_path": meta_save_path,
                "dimension": dimension
            })



    def load_plan_traces_from_export_dir(self) -> list:
        traces = []
        for fname in os.listdir(self.export_dir):
            if fname.startswith("trace_") and fname.endswith(".json"):
                fpath = os.path.join(self.export_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        loaded_dict = json.load(f)
                    trace = PlanTrace.from_dict(loaded_dict)
                    traces.append(trace)
                except Exception as e:
                    print(f"Error loading trace from {fname}: {e}")
        return traces



    def get_trace_score_stats(self, trace) -> torch.Tensor:
        sicql_q_values = []
        sicql_v_values = []
        ebt_energies = []
        ebt_uncertainties = []

        # Collect all bundles (execution steps + final)
        bundles = [step.scores for step in trace.execution_steps] + [trace.final_scores]

        for bundle in bundles:
            for dimension in self.dimensions:
                result = bundle.results.get(dimension)
                if result:
                    # Use getattr with fallback to None
                    q = getattr(result, "q_value", None)
                    v = getattr(result, "state_value", None)
                    e = getattr(result, "energy", None)
                    u = getattr(result, "uncertainty", None)

                    # Append only valid floats
                    if q is not None: 
                        sicql_q_values.append(q)
                    if v is not None: 
                        sicql_v_values.append(v)
                    if e is not None: 
                        ebt_energies.append(e)
                    if u is not None: 
                        ebt_uncertainties.append(u)

        def stats(values):
            valid = [v for v in values if v is not None]
            if not valid:
                return [0.0, 0.0, 0.0]
            return [
                float(np.mean(valid)),
                float(np.std(valid)),
                float(valid[-1]),
            ]

        # Final features vector: [q_stats, v_stats, energy_stats, uncertainty_stats]
        features = (
            stats(sicql_q_values) +
            stats(sicql_v_values) +
            stats(ebt_energies) +
            stats(ebt_uncertainties)
        )

        return torch.tensor(features, dtype=torch.float32)
