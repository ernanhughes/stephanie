# stephanie/agents/maintenance/epistemic_plan_hrm_trainer_agent.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.training.hrm_trainer import HRMTrainer # Reuse core trainer
from stephanie.scoring.model.epistemic_trace_encoder import EpistemicTraceEncoder
from stephanie.data.plan_trace import PlanTrace
# Assuming you have a way to load PlanTrace data
from stephanie.data.loaders.plan_trace_loader import load_plan_traces
import torch
from torch.utils.data import DataLoader, TensorDataset 

class EpistemicPlanHRMTrainerAgent(BaseAgent):
    """
    Agent to train the Hierarchical Reasoning Model (HRM) to evaluate
    the quality of epistemic reasoning plans (traces).
    """
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "hrm"
        # Get specific config for this HRM instance
        self.hrm_config = cfg.get("epistemic_plan_hrm", {})
        self.hrm_dimension = self.hrm_config.get("hrm_dimension", "epistemic_plan_quality")

        # Initialize the core HRM trainer logic with the specific config
        self.hrm_trainer_core = HRMTrainer(self.hrm_config, memory, logger)

        # Initialize the trace encoder
        embedding_dim = self.memory.embedding.dim # Assuming accessible
        self.trace_encoder = EpistemicTraceEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=self.hrm_config.get("trace_encoder_hidden_dim", 256),
            output_dim=self.hrm_config.get("hrm.input_dim", 128)
        ).to(self.hrm_trainer_core.device) # Move encoder to same device as HRM

        # Optimizer for the trace encoder (if it has trainable params)
        # The HRM core handles its own optimizer
        self.encoder_optimizer = torch.optim.AdamW(self.trace_encoder.parameters(), lr=self.hrm_config.get("trace_encoder_lr", 1e-4))

        self.logger.log("EpistemicPlanHRMTrainerInitialized", {
            "hrm_dimension": self.hrm_dimension,
            "hrm_config_keys": list(self.hrm_config.keys())
        })

    async def run(self, context: dict) -> dict:
        """
        Main execution logic for training the Epistemic Plan HRM.
        """
        self.logger.log("EpistemicPlanHRMTrainingStarted", { "hrm_dimension": self.hrm_dimension })

        # --- 1. Load and Prepare Training Data ---
        # The context should provide a list of PlanTrace objects
        # These traces should have their `target_epistemic_quality` populated.
        raw_traces = context.get("plan_traces", [])
        if not raw_traces:
            self.logger.log("EpistemicPlanHRMTrainingError", { "message": "No plan traces found in context." })
            return { "status": "failed", "message": "No plan traces provided." }

        # Filter traces with valid targets
        training_traces = [t for t in raw_traces if t.target_epistemic_quality is not None]
        self.logger.log("EpistemicPlanHRMTrainingDataPrepared", {
            "total_traces": len(raw_traces),
            "valid_traces": len(training_traces)
        })

        if not training_traces:
            self.logger.log("EpistemicPlanHRMTrainingError", { "message": "No plan traces with valid target quality scores." })
            return { "status": "failed", "message": "No traces with targets." }

        # --- 2. Training Loop (Simplified Outer Loop) ---
        # The core HRM trainer handles epochs/batches/loss.
        # We need to provide it with encoded inputs and targets.
        # This requires a custom data preparation step or a modified dataloader.

        # For simplicity, let's assume we modify the HRMTrainer's `train` method
        # to accept a list of (encoded_input, target) tuples or a custom dataset.
        # Or, we adapt the data here and pass it in a compatible format.

        # Let's prepare the data in the format expected by HRMTrainer
        # (Assuming HRMTrainer.train can take a 'samples' list like before,
        # but we need to encode the inputs first)
        encoded_training_samples = []
        for trace in training_traces:
            try:
                # Encode the trace using our encoder
                with torch.no_grad(): # Don't compute gradients for encoder during HRM training (simplification)
                    encoded_input_tensor = self.trace_encoder(trace, self.memory.embedding) # Shape: (1, input_dim)
                # Get the target
                target_score = trace.target_epistemic_quality
                # Package into the format HRMTrainer expects
                # Modify _create_dataloader or train method to handle tensor inputs directly
                # or convert back to a format it can re-encode (less efficient)
                # Let's assume we can pass the tensor directly or adapt the trainer slightly.
                # For now, pack it into a dict similar to before, but with tensors.
                encoded_training_samples.append({
                    "encoded_input_tensor": encoded_input_tensor.squeeze(0), # Remove batch dim for stacking later (1, input_dim) -> (input_dim)
                    "target_score": target_score
                })
            except Exception as e:
                self.logger.log("EpistemicPlanHRMTrainingDataError", {
                    "message": "Error encoding trace.",
                    "trace_id": getattr(trace, 'trace_id', 'unknown'),
                    "error": str(e)
                })
                continue

        if not encoded_training_samples:
             self.logger.log("EpistemicPlanHRMTrainingError", { "message": "No samples successfully encoded." })
             return { "status": "failed", "message": "Encoding failed for all samples." }

        # --- 3. Delegate to Core HRM Trainer ---
        # We need to adapt the call slightly.
        # Option A: Modify HRMTrainer.train to accept encoded samples.
        # Option B: Create a temporary dataloader here and pass it.
        # Option C: Call a modified internal training function.
        # Let's assume Option A: HRMTrainer.train can now handle a `pre_encoded_samples` kwarg.
        training_context = {
            "pre_encoded_samples": encoded_training_samples, # New key for this trainer
            "dimension": self.hrm_dimension # Still pass dimension for saving
            # Other HRM config is inside self.hrm_config passed to HRMTrainer init
        }

        try:
            # Execute the core training logic with encoded data
            training_result = self.hrm_trainer_core.train(**training_context) # Pass as kwargs

            self.logger.log("EpistemicPlanHRMTrainingCompleted", {
                "hrm_dimension": self.hrm_dimension,
                "result": training_result
            })

            return {
                "status": training_result.get("status", "unknown"),
                "training_result": training_result,
                "hrm_dimension": self.hrm_dimension,
            }

        except Exception as e:
            self.logger.log("EpistemicPlanHRMTrainingError", {
                "message": "Error during HRM core training execution.",
                "hrm_dimension": self.hrm_dimension,
                "error": str(e)
            })
            return {
                "status": "failed",
                "message": f"HRM training failed: {e}",
                "hrm_dimension": self.hrm_dimension
            }


# Note: This requires modifying `HRMTrainer.train` to accept and handle `pre_encoded_samples`.
# Inside `HRMTrainer.train`:
# def train(self, samples=None, dimension=None, pre_encoded_samples=None):
#     if pre_encoded_samples:
#         # Use pre_encoded_samples directly to create DataLoader
#         # inputs = torch.stack([s["encoded_input_tensor"] for s in pre_encoded_samples])
#         # targets = torch.tensor([s["target_score"] for s in pre_encoded_samples], dtype=torch.float32).unsqueeze(1)
#         # dataset = TensorDataset(inputs, targets)
#         # ... rest of DataLoader creation and training loop ...
#     elif samples:
#         # Use existing _create_dataloader(samples) logic
#     else:
#         raise ValueError("Must provide either 'samples' or 'pre_encoded_samples'")
