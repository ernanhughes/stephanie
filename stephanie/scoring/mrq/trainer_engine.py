# stephanie/scoring/mrq/trainer_engine.py
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from stephanie.models.incontext_q_model import InContextQModel
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import save_json
from stephanie.utils.metrics import EpistemicMetrics


class MRQTrainerEngine:
    def __init__(self, memory, logger, device="cpu"):
        self.memory = memory
        self.logger = logger
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim

        # Loss weights (configurable)
        self.q_weight = 1.0
        self.v_weight = 0.5
        self.pi_weight = 0.2
        self.expectile_weight = 0.8  # For V-loss
        self.entropy_weight = 0.1  # For policy regularization

        # Training parameters
        self.lr = 1e-4
        self.lr_v = 5e-5
        self.lr_pi = 3e-5
        self.epochs = 50
        self.batch_size = 32
        self.patience = 3
        self.min_delta = 0.001
        self.uncertainty_threshold = 0.3
        self.gamma = 0.95  # Discount factor

    def build_encoder(self):
        """Build text encoder for context-document fusion"""
        return TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)

    def build_predictor(self):
        """Build value predictor for MRQ compatibility"""
        return ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)

    def build_sicql_model(self, action_dim=1):
        """Build SICQL model with Q/V/π heads"""
        return InContextQModel(
            dim=self.dim,
            hdim=self.hdim,
            action_dim=action_dim,
            device=self.device,
        ).to(self.device)

    def _create_dataloader(self, encoder, samples):
        """Convert samples to PyTorch DataLoader"""
        context_embs, doc_embs, scores = [], [], []

        for idx, item in enumerate(samples):
            # Get context and document embeddings
            context = item.get("title", "")
            context_emb = self.memory.embedding.get_or_create(context)

            # Process both A and B samples
            for side in ["a", "b"]:
                doc_text = item[f"output_{side}"]
                doc_emb = self.memory.embedding.get_or_create(doc_text)

                # Store data
                context_embs.append(torch.tensor(context_emb))
                doc_embs.append(torch.tensor(doc_emb))
                scores.append(float(item[f"value_{side}"]))

        # Convert to tensors
        context_tensors = torch.stack(context_embs).to(self.device)
        doc_tensors = torch.stack(doc_embs).to(self.device)
        score_tensors = torch.tensor(scores).float().to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(context_tensors, doc_tensors, score_tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _compute_losses(self, outputs, scores, next_values=None):
        """Compute all SICQL losses"""
        losses = {}

        # Q-loss: MSE vs LLM scores
        if next_values is not None:
            # Use Bellman target with gamma
            q_target = scores + self.gamma * next_values
        else:
            q_target = scores

        losses["q"] = nn.MSELoss()(outputs["q_value"].squeeze(), q_target)

        # V-loss: Expectile regression between Q and V
        with torch.no_grad():
            advantage = (
                outputs["q_value"].detach() - outputs["state_value"].detach()
            )
            expectile_mask = (advantage > 0).float()

        losses["v"] = torch.mean(
            self.expectile_weight
            * torch.abs(outputs["state_value"].squeeze() - q_target.detach())
            * expectile_mask
        )

        # Policy loss: Advantage-weighted regression (AWR)
        action_probs = F.softmax(outputs["action_probs"], dim=-1)
        advantage = (outputs["q_value"] - outputs["state_value"]).detach()
        losses["pi"] = -torch.mean(
            torch.log(action_probs) * advantage.unsqueeze(-1)
        )

        # Entropy regularization
        dist = torch.distributions.Categorical(
            logits=outputs["action_probs"]
        )
        losses["entropy"] = -self.entropy_weight * dist.entropy().mean()

        # Total loss
        losses["total"] = (
            self.q_weight * losses["q"]
            + self.v_weight * losses["v"]
            + self.pi_weight * losses["pi"]
            + losses["entropy"]
        )

        return losses

    def _train_epoch(self, model, dataloader, optimizers):
        """Train for one epoch"""
        model.train()
        epoch_losses = defaultdict(list)

        for context_emb, doc_emb, scores in tqdm(dataloader, desc="Training"):
            # Forward pass
            outputs = model(context_emb, doc_emb)

            # Compute losses
            losses = self._compute_losses(outputs, scores)

            # Backward pass
            optimizers["q"].zero_grad()
            optimizers["v"].zero_grad()
            optimizers["pi"].zero_grad()

            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # Update weights
            optimizers["q"].step()
            optimizers["v"].step()
            optimizers["pi"].step()

            # Record losses
            for k, v in losses.items():
                epoch_losses[k].append(v.item())

        return {k: np.mean(v) for k, v in epoch_losses.items()}

    def _validate(self, model, val_loader):
        """Validation phase"""
        model.eval()
        val_losses = defaultdict(list)

        with torch.no_grad():
            for context_emb, doc_emb, scores in val_loader:
                outputs = model(context_emb, doc_emb)
                losses = self._compute_losses(outputs, scores)

                for k, v in losses.items():
                    val_losses[k].append(v.item())

        return {f"val_{k}": np.mean(v) for k, v in val_losses.items()}

    def _setup_optimizers(self, model):
        """Initialize optimizers for all heads"""
        return {
            "q": optim.Adam(model.q_head.parameters(), lr=self.lr),
            "v": optim.Adam(model.v_head.parameters(), lr=self.lr_v),
            "pi": optim.Adam(model.pi_head.parameters(), lr=self.lr_pi),
        }

    def _setup_schedulers(self, optimizers):
        """Initialize learning rate schedulers"""
        return {
            "q": ReduceLROnPlateau(
                optimizers["q"], mode="min", factor=0.5, patience=2
            ),
            "v": ReduceLROnPlateau(
                optimizers["v"], mode="min", factor=0.5, patience=2
            ),
            "pi": ReduceLROnPlateau(
                optimizers["pi"], mode="min", factor=0.5, patience=2
            ),
        }

    def train_sicql(
        self, model, dataloader, val_loader=None, output_dir="models"
    ):
        """Main training loop with SICQL enhancements"""
        # Setup optimizers and schedulers
        optimizers = self._setup_optimizers(model)
        schedulers = self._setup_schedulers(optimizers)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            train_losses = self._train_epoch(model, dataloader, optimizers)

            # Validation phase
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate(model, val_loader)
                val_loss = val_metrics["val_total"]
            else:
                val_loss = train_losses["total"]

            # Logging
            self.logger.log(
                "SICQLTrainingEpoch",
                {
                    "epoch": epoch + 1,
                    "train_loss": train_losses["total"],
                    "val_loss": val_loss,
                    "q_loss": train_losses["q"],
                    "v_loss": train_losses["v"],
                    "pi_loss": train_losses["pi"],
                    "entropy": train_losses["entropy"],
                    "lr": optimizers["q"].param_groups[0]["lr"],
                },
            )

            # Early stopping
            if val_loss < best_loss - self.min_delta:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                self.logger.log(
                    "SICQLEarlyStopping",
                    {"epoch": epoch + 1, "best_loss": best_loss},
                )    
                break

            # Update learning rates
            for name in ["q", "v", "pi"]:
                if val_loader:
                    schedulers[name].step(val_loss)
                else:
                    schedulers[name].step(train_losses["total"])

        # Final model save
        torch.save(model.state_dict(), f"{output_dir}/final_model.pt")
        self.logger.log("SICQLTrainingComplete", {"best_loss": best_loss})

        return model

    def detect_epistemic_gaps(self, model, dataloader):
        """Identify areas where model uncertainty is high"""
        model.eval()
        gaps = []

        with torch.no_grad():
            for context_emb, doc_emb, scores in dataloader:
                outputs = model(context_emb, doc_emb)

                # Compute uncertainty
                uncertainties = EpistemicMetrics.compute_uncertainty(
                    outputs["q_value"], outputs["state_value"]
                )

                # Find high-uncertainty samples
                for i in range(len(uncertainties)):
                    if uncertainties[i] > self.uncertainty_threshold:
                        gaps.append(
                            {
                                "sample_idx": i,
                                "uncertainty": uncertainties[i].item(),
                                "predicted_score": outputs["q_value"][
                                    i
                                ].item(),
                                "llm_score": scores[i].item(),
                                "document_text": doc_emb[
                                    i
                                ].tolist(),  # Convert to list for JSON
                            }
                        )

        # Log gaps
        for gap in gaps:
            EpistemicMetrics.log_epistemic_gap(gap)

        return gaps

    def train_all(self, contrast_pairs, cfg=None):
        """Train models for all dimensions"""
        if cfg:
            self._update_config(cfg)

        trained_models = {}
        trained_encoders = {}
        regression_tuners = {}

        # Group pairs by dimension
        pairs_by_dim = defaultdict(list)
        for item in contrast_pairs:
            pairs_by_dim[item["dimension"]].append(item)

        # Train for each dimension
        for dim, samples in pairs_by_dim.items():
            self.logger.log(
                "SICQLTrainingDimension",
                {"dimension": dim, "sample_count": len(samples)},
            )

            # Build dataloader
            dataloader = self._create_dataloader(self.build_encoder(), samples)

            # Initialize model

            use_sicql = self.cfg.get("use_sicql_style", False)

            if use_sicql:
                model = self.build_sicql_model(action_dim=1)
            else:
                encoder = self.build_encoder()
                predictor = self.build_predictor()
                from stephanie.scoring.mrq.model import MRQModel
                model = MRQModel(encoder, predictor, self.memory.embedding, device=self.device)

            # Create output directory
            output_dir = f"{self.cfg.get('model_path', 'models')}/{dim}"
            os.makedirs(output_dir, exist_ok=True)

            # Train
            if use_sicql:
                trained_model = self.train_sicql(
                    model, dataloader, output_dir=output_dir
                )
                trained_models[dim] = trained_model.state_dict()
                trained_encoders[dim] = trained_model.encoder.state_dict()
                torch.save(
                    trained_model.q_head.state_dict(), f"{output_dir}/q_head.pt"
                )
                torch.save(
                    trained_model.v_head.state_dict(), f"{output_dir}/v_head.pt"
                )
                torch.save(
                    trained_model.pi_head.state_dict(), f"{output_dir}/pi_head.pt"
                )
            else:
                model = self._train_mrq(model, dataloader, output_dir=output_dir)  
                trained_model = model.predictor
                trained_models[dim] = model.predictor.state_dict()
                trained_encoders[dim] = model.encoder.state_dict()

            # Save metadata
            meta = {
                "dim": self.dim,
                "hdim": self.hdim,
                "dimension": dim,
                "model_type": "sicql" if use_sicql else "mrq",
                "target_type": self.cfg.get("target_type", "document"),
                "embedding_type": self.cfg.get("embedding_type", "hnet"),
                "version": self.cfg.get("model_version", "v1"),
                "training_params": {
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.lr,
                    "gamma": self.gamma,
                },
            }
            save_json(meta, f"{output_dir}/meta.json")

            # Build regression tuner
            regression_tuners[dim] = RegressionTuner.from_dataloader(
                dataloader,
                model=trained_model,
                dimension=dim,
                logger=self.logger,
            )

        return trained_encoders, trained_models, regression_tuners

    def _update_config(self, cfg):
        """Update training parameters from config"""
        self.cfg = cfg
        self.lr = cfg.get("lr", self.lr)
        self.lr_v = cfg.get("lr_v", self.lr_v)
        self.lr_pi = cfg.get("lr_pi", self.lr_pi)
        self.epochs = cfg.get("epochs", self.epochs)
        self.batch_size = cfg.get("batch_size", self.batch_size)
        self.patience = cfg.get("patience", self.patience)
        self.min_delta = cfg.get("min_delta", self.min_delta)
        self.uncertainty_threshold = cfg.get(
            "uncertainty_threshold", self.uncertainty_threshold
        )
        self.gamma = cfg.get("gamma", self.gamma)
        self.q_weight = cfg.get("q_weight", self.q_weight)
        self.v_weight = cfg.get("v_weight", self.v_weight)
        self.pi_weight = cfg.get("pi_weight", self.pi_weight)
        self.expectile_weight = cfg.get(
            "expectile_weight", self.expectile_weight
        )
        self.entropy_weight = cfg.get("entropy_weight", self.entropy_weight)

    def _train_mrq(self, model, dataloader, output_dir=None):
        """Train a standard MRQModel (Q only) using MSE"""
        model.train_mode()  # ✅ FIXED

        optimizer = optim.Adam(
            list(model.encoder.parameters()) + list(model.predictor.parameters()),
            lr=self.lr
        )
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            total_loss = 0
            count = 0

            for context_emb, doc_emb, scores in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()

                # Forward pass
                zsa = model.encoder(context_emb, doc_emb)
                q_pred = model.predictor(zsa).squeeze()

                # MSE loss
                loss = F.mse_loss(q_pred, scores)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 0.5)

                optimizer.step()

                total_loss += loss.item() * context_emb.size(0)
                count += context_emb.size(0)

            avg_loss = total_loss / count

            self.logger.log("MRQTrainingEpoch", {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if avg_loss < best_loss - self.min_delta:
                best_loss = avg_loss
                patience_counter = 0
                if output_dir:
                    torch.save(model.encoder.state_dict(), f"{output_dir}/encoder.pt")
                    torch.save(model.predictor.state_dict(), f"{output_dir}/predictor.pt")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                self.logger.log("MRQEarlyStopping", {
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                })
                break

        self.logger.log("MRQTrainingComplete", {"best_loss": best_loss})
        return model
