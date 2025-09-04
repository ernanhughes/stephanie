# stephanie/training/hrm_trainer.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW  # As recommended by HRM paper
from torch.utils.data import DataLoader, TensorDataset

from stephanie.scoring.model.hrm_model import HRMModel  # Import the model
from stephanie.scoring.training.base_trainer import \
    BaseTrainer  # Assuming this exists or adapt


class HRMTrainer(BaseTrainer): 
    """
    Trainer Agent for the Hierarchical Reasoning Model (HRM).
    Integrates with Stephanie's training framework.
    """
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        
        # --- HRM Specific Config ---
        self.model_type = "hrm"
        self.embedding_type = self.memory.embedding.name
        embedding_dim = memory.embedding.dim
        self.input_dim = embedding_dim * 2
        self.h_dim = cfg.get("hrm.h_dim", 256)
        self.l_dim = cfg.get("hrm.l_dim", 128)
        self.output_dim = cfg.get("hrm.output_dim", 1) # 1 for score prediction
        self.n_cycles = cfg.get("hrm.n_cycles", 4)
        self.t_steps = cfg.get("hrm.t_steps", 4)
        self.lr = cfg.get("hrm.lr", 1e-4)
        self.epochs = cfg.get("hrm.epochs", 10)
        self.batch_size = cfg.get("hrm.batch_size", 32)
        self.apply_sigmoid = cfg.get("hrm.apply_sigmoid", True) 
        self.scaler_p_lo = cfg.get("hrm.target_scale_p_lo", 10.0)
        self.scaler_p_hi = cfg.get("hrm.target_scale_p_hi", 90.0) 
        self._scaler_stats = {}  

        # Device setup (inherited or set)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the HRM model
        hrm_cfg = {
            "hrm.input_dim": self.input_dim,
            "hrm.h_dim": self.h_dim,
            "hrm.l_dim": self.l_dim,
            "hrm.output_dim": self.output_dim,
            "hrm.n_cycles": self.n_cycles,
            "hrm.t_steps": self.t_steps,
        }
        # Assuming HRMModel is correctly imported
        self.hrm_model = HRMModel(hrm_cfg, logger=self.logger).to(self.device)
        
        # Optimizer (AdamW as recommended)
        self.optimizer = AdamW(self.hrm_model.parameters(), lr=self.lr)
        
        # Loss function (MSE for regression, e.g., predicting a score)
        # Can be made configurable (e.g., CrossEntropy for classification)
        self.criterion = nn.MSELoss() 

        self.logger.log("HRMTrainerInitialized", {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "h_dim": self.h_dim,
            "l_dim": self.l_dim,
            "output_dim": self.output_dim,
            "n_cycles": self.n_cycles,
            "t_steps": self.t_steps,
            "lr": self.lr,
            "device": str(self.device)
        })

    def train (self, samples, dimension) -> dict:
        """
        Main training loop.
        Expects 'training_data' in context, or loads it via _create_dataloader.
        """
        self.logger.log("HRMTrainingStarted", {"epochs": self.epochs})

        dataloader = self._create_dataloader(samples, dimension)
        if dataloader is None:
             self.logger.log("HRMTrainingError", {"message": "Dataloader creation failed or insufficient samples."})
             return {"status": "failed", "message": "Dataloader creation failed."}

        # 2. Training Loop
        for epoch in range(self.epochs):
            epoch_loss, num_batches = 0.0, 0
            for _, (x_batch, y_batch) in enumerate(dataloader):
                # Move data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                y_pred, _ = self.hrm_model(x_batch)  # (B,1) expected
                if self.apply_sigmoid:               # NEW
                    y_pred = torch.sigmoid(y_pred)   # keep outputs in [0,1]

                                # Compute loss
                # Ensure y_batch has the correct shape for the loss function
                # e.g., if output_dim=1, y_batch should be (B, 1) or (B,)
                # MSELoss expects same shape for pred and target
                loss = self.criterion(y_pred, y_batch)
                # Backward pass (One-step gradient approximation)
                # PyTorch's autograd handles this naturally for the looped architecture
                # as long as we don't unroll the entire N*T steps explicitly in the graph
                # and use the final loss.
                loss.backward()

                # Update parameters
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1                
                
                self.logger.log("HRMTrainingBatch", {"epoch": epoch, "loss": loss.item()})

            # Log average epoch loss
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.logger.log("HRMTrainingEpoch", {"epoch": epoch, "avg_loss": avg_epoch_loss})

        # 3. Save Model
        self._save_model(dimension)
        
        self.logger.log("HRMTrainingCompleted", {"final_avg_loss": avg_epoch_loss})
        return {"status": "trained", "final_loss": avg_epoch_loss}

    def _create_dataloader(self, samples, dimension):  # CHANGED signature
        """
        Build tensors and scale targets to [0,1] using robust percentiles (p10..p90).
        """
        # 1) Collect and fit scaler on raw target scores
        raw_targets = []
        pre_filtered = []
        for s in samples:
            goal_text = s.get("goal_text", "")
            scorable_text = s.get("scorable_text", "")
            target_value = s.get("target_score", s.get("score", None))
            if not goal_text or not scorable_text or target_value is None:
                continue
            pre_filtered.append((goal_text, scorable_text, float(target_value)))
            raw_targets.append(float(target_value))

        if len(pre_filtered) < self.min_samples:
            self.logger.log("HRMDataError", {"message": f"Insufficient raw samples: {len(pre_filtered)} < {self.min_samples}"})
            return None

        # Robust percentiles for stability
        lo = float(np.percentile(raw_targets, self.scaler_p_lo)) if raw_targets else 0.0
        hi = float(np.percentile(raw_targets, self.scaler_p_hi)) if raw_targets else 1.0
        if hi - lo < 1e-9:
            hi = lo + 1.0

        self._scaler_stats[dimension] = {"lo": lo, "hi": hi}  # store for meta/logs
        self.logger.log("HRMTargetScaling", {
            "dimension": dimension, "p_lo": self.scaler_p_lo, "p_hi": self.scaler_p_hi,
            "lo": lo, "hi": hi, "n": len(raw_targets)
        })

        def _norm(v: float) -> float:
            x = (v - lo) / (hi - lo)
            return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

        # 2) Encode inputs and normalized targets
        valid_samples = []
        for goal_text, scorable_text, target_value in pre_filtered:
            try:
                ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal_text), dtype=torch.float32)
                doc_emb = torch.tensor(self.memory.embedding.get_or_create(scorable_text), dtype=torch.float32)
                input_tensor = torch.cat([ctx_emb, doc_emb], dim=-1)
                target_tensor = torch.tensor([_norm(target_value)], dtype=torch.float32)  # normalized → [0,1]
                valid_samples.append((input_tensor, target_tensor))
            except Exception as e:
                self.logger.log("HRMDataError", {"error": str(e), "sample_goal_preview": goal_text[:80]})
                continue

        if len(valid_samples) < self.min_samples:
            self.logger.log("HRMDataError", {"message": f"Insufficient valid samples: {len(valid_samples)} < {self.min_samples}"})
            return None

        inputs, targets = zip(*valid_samples)
        dataset = TensorDataset(torch.stack(inputs), torch.stack(targets))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.logger.log("HRMDataLoaderCreated", {
            "dimension": dimension,
            "num_samples": len(valid_samples),
            "num_batches": len(dataloader)
        })
        return dataloader

    def _save_model(self, dimension: str):
        locator = self.get_locator(dimension)
        torch.save(self.hrm_model.state_dict(), locator.model_file(suffix="_hrm.pt"))

        meta = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "h_dim": self.h_dim,
            "l_dim": self.l_dim,
            "output_dim": self.output_dim,
            "n_cycles": self.n_cycles,
            "t_steps": self.t_steps,
            "lr": self.lr,
            "epochs": self.epochs,
            "apply_sigmoid": self.apply_sigmoid,                 # NEW
            "target_scale": {                                   # NEW
                "method": "robust_minmax",
                "p_lo": self.scaler_p_lo,
                "p_hi": self.scaler_p_hi,
                "lo": self._scaler_stats.get(dimension, {}).get("lo"),
                "hi": self._scaler_stats.get(dimension, {}).get("hi"),
            },
        }
        self._save_meta_file(meta, dimension)
        self.logger.log("HRMModelSaved", {"path": locator.base_path, "dimension": dimension})
