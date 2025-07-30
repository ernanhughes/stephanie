# stephanie/training/hrm_trainer.py
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
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        
        # --- HRM Specific Config ---
        self.model_type = "hrm"
        self.embedding_type = memory.embedding.type
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
        # --------------------------

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

        dataloader = self._create_dataloader(samples)
        if dataloader is None:
             self.logger.log("HRMTrainingError", {"message": "Dataloader creation failed or insufficient samples."})
             return {"status": "failed", "message": "Dataloader creation failed."}

        # 2. Training Loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for _, (x_batch, y_batch) in enumerate(dataloader):
                # Move data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                y_pred, intermediate_states = self.hrm_model(x_batch) # (B, output_dim)

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
                
                # Optional: Log batch loss
                # self.logger.log("HRMTrainingBatch", {"epoch": epoch, "batch": batch_idx, "loss": loss.item()})

            # Log average epoch loss
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.logger.log("HRMTrainingEpoch", {"epoch": epoch, "avg_loss": avg_epoch_loss})

        # 3. Save Model
        self._save_model(dimension)
        
        self.logger.log("HRMTrainingCompleted", {"final_avg_loss": avg_epoch_loss})
        return {"status": "trained", "final_loss": avg_epoch_loss}

    def _create_dataloader(self, samples):
        """
        Creates a DataLoader for HRM training.
        Assumes samples contain context_text, document_text, and a target_score.
        This is a basic example. You might need more complex logic based on your
        specific task (e.g., predicting next step in a sequence).
        """
        valid_samples = []
        for s in samples:
            ctx_text = s.get("context_text", "") # Or goal_text
            doc_text = s.get("document_text", "") # Or scorable.text
            # Target for HRM training. This is crucial.
            # Example: Predicting a score (like SICQL Q-value) or a derived metric.
            target_value = s.get("target_score", s.get("score", None)) 
            
            # Example: Using SICQL score as target
            # target_value = s.get("sicql_q_value", None) 

            if not ctx_text or not doc_text or target_value is None:
                continue # Skip invalid samples

            try:
                ctx_emb = torch.tensor(self.memory.embedding.get_or_create(ctx_text), dtype=torch.float32)
                doc_emb = torch.tensor(self.memory.embedding.get_or_create(doc_text), dtype=torch.float32)
                target_tensor = torch.tensor([target_value], dtype=torch.float32) # Shape (1,) for MSE with output_dim=1
                
                # Input to HRM: Concatenated embeddings
                input_tensor = torch.cat([ctx_emb, doc_emb], dim=-1) # Shape (input_dim,)
                
                valid_samples.append((input_tensor, target_tensor))
            except Exception as e:
                self.logger.log("HRMDataError", {"error": str(e), "sample_id": s.get("id", "unknown")})
                continue

        if len(valid_samples) < self.min_samples: # Assuming min_samples is in cfg or BaseTrainer
            self.logger.log("HRMDataError", {"message": f"Insufficient valid samples: {len(valid_samples)} < {self.min_samples}"})
            return None

        # Create TensorDataset and DataLoader
        inputs, targets = zip(*valid_samples)
        dataset = TensorDataset(torch.stack(inputs), torch.stack(targets))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.logger.log("HRMDataLoaderCreated", {"num_samples": len(valid_samples), "num_batches": len(dataloader)})
        return dataloader

    def _save_model(self, dimension: str):
        """Saves the trained HRM model components using the Locator."""
        locator = self.get_locator(dimension) # Assuming BaseTrainer provides this
        
        # Save model state dict
        torch.save(self.hrm_model.state_dict(), locator.model_file(suffix="_hrm.pt"))
        
        # Save individual components if needed (optional, but matches SICQL pattern)
        # torch.save(self.hrm_model.input_projector.state_dict(), locator.model_file(suffix="_input.pt"))
        # torch.save(self.hrm_model.l_module.state_dict(), locator.model_file(suffix="_l.pt"))
        # torch.save(self.hrm_model.h_module.state_dict(), locator.model_file(suffix="_h.pt"))
        # torch.save(self.hrm_model.output_projector.state_dict(), locator.model_file(suffix="_output.pt"))
        
        # Save configuration
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
        }
        self._save_meta_file(meta, dimension) # Assuming this method exists in BaseTrainer
        
        self.logger.log("HRMModelSaved", {"path": locator.base_path})
