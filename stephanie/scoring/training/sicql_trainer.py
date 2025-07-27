# stephanie/agents/maintenance/sicql_trainer.py
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.models.model_version import ModelVersionORM
from stephanie.models.training_stats import TrainingStatsORM
from stephanie.scoring.model.in_context_q import InContextQModel
from stephanie.scoring.model.policy_head import PolicyHead
from stephanie.scoring.model.q_head import QHead
from stephanie.scoring.model.v_head import VHead
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class SICQLTrainer(BaseTrainer):

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.embedding_type = self.memory.embedding.type
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim
        self.root_dir = cfg.get("model_path", "models")
        self.dimension = cfg.get("dimension", "alignment")
        self.embedding_type = cfg.get("embedding_type", "hnet")
        self.model_type = "sicql"
        self.target_type = cfg.get("target_type", "document")
        self.version = cfg.get("model_version", "v1")

        # Device management
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Training configuration
        self._init_config(cfg)

        # Track training state
        self.best_loss = float("inf")
        self.early_stop_counter = 0
        self.models = {}
        self.tuners = {}
        self._load_tuners()

        # Log initialization
        self.logger.log(
            "SICQLTrainerInitialized",
            {
                "dimension": self.cfg.get("dimension", "alignment"),
                "embedding_type": self.cfg.get("embedding_type", "hnet"),
                "use_gild": self.use_gild,
                "use_qmax": self.use_qmax,
                "device": str(self.device),
            },
        )
 

    def _init_config(self, cfg):
        """Initialize training parameters from config"""
        self.use_tuner = cfg.get("use_tuner", True)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.batch_size = cfg.get("batch_size", 32)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("lr", 1e-4)
        self.gamma = cfg.get("gamma", 0.95)  # Discount factor
        self.beta = cfg.get("beta", 1.0)  # Policy temperature
        self.entropy_weight = cfg.get("entropy_weight", 0.01)
        self.dimensions = cfg.get("dimensions", [])
        self.min_samples = cfg.get("min_samples", 10)
        self.expectile_tau = cfg.get("expectile_tau", 0.7)  # For V-head
        self.use_gild = cfg.get("use_gild", True)
        self.use_qmax = cfg.get("use_qmax", True)
        self.scorer_map = ["ebt", "svm", "mrq"]  # Policy head mapping

    def _load_tuners(self):
        """Load regression tuners for each dimension"""
        for dim in self.dimensions:
            tuner_path = super().get_locator(dim).tuner_file()
            if os.path.exists(tuner_path):
                self.tuners[dim] = RegressionTuner(dimension=dim)
                self.tuners[dim].load(tuner_path)
            else:
                self.tuners[dim] = None
                self.logger.log(
                    "TunerMissing", {"dimension": dim, "path": tuner_path}
                )

    def _build_model(self, dimension):
        """Build or load SICQL model"""
        locator = super().get_locator(dimension)
        if locator.model_exists():
            # Load existing model
            encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
            q_head = QHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            v_head = VHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            pi_head = PolicyHead(
                zsa_dim=self.dim, hdim=self.hdim, num_actions=3
            ).to(self.device)

            # Load weights
            encoder.load_state_dict(
                torch.load(locator.encoder_file(), map_location=self.device)
            )
            q_head.load_state_dict(
                torch.load(locator.q_head_file(), map_location=self.device)
            )
            v_head.load_state_dict(
                torch.load(locator.v_head_file(), map_location=self.device)
            )
            pi_head.load_state_dict(
                torch.load(locator.pi_head_file(), map_location=self.device)
            )

            # Build model
            sicql_model = InContextQModel(
                encoder=encoder,
                q_head=q_head,
                v_head=v_head,
                pi_head=pi_head,
                embedding_store=self.memory.embedding,
                device=self.device,
            )
            return sicql_model

        # Build new model
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim

        encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
        q_head = QHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        v_head = VHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        pi_head = PolicyHead(
            zsa_dim=self.dim, hdim=self.hdim, num_actions=3
        ).to(self.device)

        return InContextQModel(
            encoder=encoder,
            q_head=q_head,
            v_head=v_head,
            pi_head=pi_head,
            embedding_store=self.memory.embedding,
            device=self.device,
        )

    def _train_epoch(self, model, dataloader):
        """Train for one epoch with all heads"""
        model.train()
        total_q_loss = 0.0
        total_v_loss = 0.0
        total_pi_loss = 0.0
        count = 0

        for ctx_emb, doc_emb, scores in tqdm(dataloader, desc="Training"):
            ctx_emb = ctx_emb.to(self.device)
            doc_emb = doc_emb.to(self.device)
            scores = scores.to(self.device)

            outputs = model(ctx_emb, doc_emb)

            q_loss = F.mse_loss(outputs["q_value"], scores)

            v_loss = (
                self._expectile_loss(
                    scores - outputs["state_value"], tau=self.expectile_tau
                )
                if self.use_qmax
                else torch.tensor(0.0, device=self.device)
            )

            pi_loss = torch.tensor(0.0, device=self.device)
            if self.use_gild and "action_logits" in outputs:
                advantage = (
                    outputs["q_value"] - outputs["state_value"]
                ).detach()
                weights = torch.exp(self.beta * advantage)
                weights = weights / weights.sum()

                # Corrected reshape
                weights = weights.unsqueeze(-1)  # Ensure (batch_size, 1)

                log_probs = F.log_softmax(outputs["action_logits"], dim=-1)
                pi_loss = -(log_probs * weights).mean()

                # Optional entropy regularization
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
                pi_loss += self.entropy_weight * entropy

            loss = (
                q_loss * self.cfg.get("q_weight", 1.0)
                + v_loss * self.cfg.get("v_weight", 0.5)
                + pi_loss * self.cfg.get("pi_weight", 0.3)
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            self.optimizer.step()

            total_q_loss += q_loss.item() * ctx_emb.size(0)
            total_v_loss += v_loss.item() * ctx_emb.size(0)
            total_pi_loss += pi_loss.item() * ctx_emb.size(0)
            count += ctx_emb.size(0)

        avg_q = total_q_loss / count
        avg_v = total_v_loss / count
        avg_pi = total_pi_loss / count

        if self.use_qmax:
            self.scheduler["q"].step(avg_q)
        if self.use_gild:
            self.scheduler["pi"].step(avg_pi)

        return {"q": avg_q, "v": avg_v, "pi": avg_pi, "total": loss.item()}

    def _expectile_loss(self, diff, tau=0.7):
        """Compute expectile loss for V-head"""
        return torch.where(
            diff > 0, tau * diff.pow(2), (1 - tau) * diff.pow(2)
        ).mean()

    def _should_stop_early(self, current_avg):
        """Check for early stopping"""
        if not self.use_early_stopping:
            return False

        if current_avg < self.best_loss - self.early_stopping_min_delta:
            self.best_loss = current_avg
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        return self.early_stop_counter >= self.early_stopping_patience

    def _save_model(self, model, dimension, stats):
        locator = super().get_locator(dimension)
        """Save model components with metadata"""
        # Save each component
        torch.save(model.encoder.state_dict(), locator.encoder_file())
        torch.save(model.q_head.state_dict(), locator.q_head_file())
        torch.save(model.v_head.state_dict(), locator.v_head_file())
        torch.save(model.pi_head.state_dict(), locator.pi_head_file())

        # Calculate policy metrics
        policy_logits = model.pi_head.weight.data.mean(dim=0).tolist()
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1).tolist()
        policy_entropy = -torch.sum(
            policy_probs * torch.log(torch.tensor(policy_probs) + 1e-8)
        ).item()

        # Build metadata
        meta = {
            "dim": self.dim,
            "hdim": self.hdim,
            "dimension": dimension,
            "version": self.cfg.get("model_version", "v1"),
            "avg_q_loss": stats.get("avg_q_loss", 0.0),
            "avg_v_loss": stats.get("avg_v_loss", 0.0),
            "avg_pi_loss": stats.get("avg_pi_loss", 0.0),
            "policy_logits": policy_logits,
            "policy_probs": policy_probs,
            "policy_entropy": policy_entropy,
            "policy_stability": max(policy_probs),
            "device": str(self.device),
            "embedding_type": self.cfg.get("embedding_type", "hnet"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Save metadata
        with open(locator.meta_file(), "w") as f:
            json.dump(meta, f)

        # Save tuner if available
        if dimension in self.tuners and self.tuners[dimension]:
            self.tuners[dimension].save(locator.tuner_file())

        # Save model version
        model_version = ModelVersionORM(**meta)
        self.memory.session.add(model_version)
        self.memory.session.commit()

        return meta

    def _log_training_stats(self, dim, meta):
        """Log training stats to database"""
        training_stats = TrainingStatsORM(
            model_type="sicql",
            target_type=self.cfg.get("target_type", "document"),
            dimension=dim,
            version=meta["version"],
            avg_q_loss=meta["avg_q_loss"],
            avg_v_loss=meta["avg_v_loss"],
            avg_pi_loss=meta["avg_pi_loss"],
            policy_entropy=meta["policy_entropy"],
            policy_stability=meta["policy_stability"],
            performance=meta["avg_q_loss"],
        )
        self.memory.session.add(training_stats)
        self.memory.session.commit()

    def _validate_tensor(self, tensor, name):
        """Validate tensor before use"""
        if tensor is None:
            self.logger.log(
                "InvalidTensor",
                {"tensor_name": name, "reason": "tensor_is_none"},
            )
            return False

        if torch.isnan(tensor).any():
            self.logger.log(
                "NaNInTensor", {"tensor_name": name, "tensor": tensor.tolist()}
            )
            return False

        return True

    def _calculate_policy_logits(self, model):
        """Calculate policy logits from policy head weights"""
        with torch.no_grad():
            policy_weights = model.pi_head.get_policy_weights()
            policy_probs = F.softmax(policy_weights, dim=-1)
            return policy_probs.tolist()

    def _calculate_policy_stability(self, policy_logits):
        """Calculate policy stability from logits"""
        if not policy_logits:
            return 0.0
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1)
        return policy_probs.max().item()

    def _calculate_policy_entropy(self, policy_logits):
        """Calculate policy entropy for versioning"""
        if not policy_logits:
            return 0.0
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1)
        return (
            -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=-1)
            .mean()
            .item()
        )

    def train(self, samples, dim):
        """
        Train SICQL model for a dimension
        Args:
            samples: List of training samples
            dim: Dimension to train
        Returns:
            Training statistics and model
        """
        self.logger.log("DimensionTrainingStarted", {"dimension": dim})

        # Prepare data
        dataloader = super()._create_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data", "dimension": dim}

        # Build model
        model = self._build_model(dim)
        model.train()

        # Optimizer for all heads
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = {
            "q": ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            ),
            "v": ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            ),
            "pi": ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            ),
        }

        # Training stats
        stats = {
            "dimension": dim,
            "q_losses": [],
            "v_losses": [],
            "pi_losses": [],
            "policy_entropies": [],
            "avg_q_loss": 0.0,
            "avg_v_loss": 0.0,
            "avg_pi_loss": 0.0,
            "policy_entropy": 0.0,
            "policy_stability": 0.0,
        }

        # Training loop
        for epoch in range(self.epochs):
            epoch_stats = self._train_epoch(model, dataloader)
            stats["q_losses"].append(epoch_stats["q"])
            stats["v_losses"].append(epoch_stats["v"])
            stats["pi_losses"].append(epoch_stats["pi"])

            # Calculate policy entropy
            policy_logits = self._calculate_policy_logits(model)
            policy_entropy = self._calculate_policy_entropy(policy_logits)
            stats["policy_entropies"].append(policy_entropy)

            # Early stopping check
            if self._should_stop_early(stats["q_losses"][-1]):
                self.logger.log(
                    "EarlyStopping",
                    {
                        "dimension": dim,
                        "epoch": epoch + 1,
                        "best_loss": self.best_loss,
                    },
                )
                break

        # Final stats
        stats["avg_q_loss"] = np.mean(stats["q_losses"])
        stats["avg_v_loss"] = np.mean(stats["v_losses"])
        stats["avg_pi_loss"] = np.mean(stats["pi_losses"])
        stats["policy_entropy"] = np.mean(stats["policy_entropies"])
        stats["policy_stability"] = (
            max(stats["policy_entropies"])
            if stats["policy_entropies"]
            else 0.0
        )

        # Save model
        meta = self._save_model(model, dim, stats)
        stats.update(meta)

        # Log to database
        self._log_training_stats(dim, meta)

        self.logger.log(
            "DimensionTrainingComplete",
            {
                "dimension": dim,
                "final_q_loss": stats["avg_q_loss"],
                "final_v_loss": stats["avg_v_loss"],
                "final_pi_loss": stats["avg_pi_loss"],
            },
        )

        # Cache model
        self.models[dim] = model
        return stats

    def _log_training_stats(self, dim, meta):
        """Log training stats to database"""
        training_stats = TrainingStatsORM(
            model_type="sicql",
            target_type=self.cfg.get("target_type", "document"),
            dimension=dim,
            version=meta["version"],
            embedding_type=self.embedding_type,
            avg_q_loss=meta["avg_q_loss"],
            avg_v_loss=meta["avg_v_loss"],
            avg_pi_loss=meta["avg_pi_loss"],
            policy_entropy=meta.get("policy_entropy", 0.0),
            policy_stability=meta.get("policy_stability", 0.0),
        )
        self.memory.session.add(training_stats)
        self.memory.session.commit()

    def _train_sicql(self, model, dataloader, output_dir):
        """Train SICQL model with all heads"""
        model.train()
        best_loss = float("inf")
        patience_counter = 0

        # Build optimizers
        optimizers = {
            "encoder": optim.Adam(model.encoder.parameters(), lr=self.lr),
            "q_head": optim.Adam(model.q_head.parameters(), lr=self.lr),
            "v_head": optim.Adam(model.v_head.parameters(), lr=self.lr),
            "pi_head": optim.Adam(model.pi_head.parameters(), lr=self.lr),
        }

        # Build schedulers
        schedulers = {
            "encoder": ReduceLROnPlateau(
                optimizers["encoder"], mode="min", factor=0.5, patience=2
            ),
            "q_head": ReduceLROnPlateau(
                optimizers["q_head"], mode="min", factor=0.5, patience=2
            ),
            "v_head": ReduceLROnPlateau(
                optimizers["v_head"], mode="min", factor=0.5, patience=2
            ),
            "pi_head": ReduceLROnPlateau(
                optimizers["pi_head"], mode="min", factor=0.5, patience=2
            ),
        }

        # Training loop
        for epoch in range(self.epochs):
            total_q_loss = 0.0
            total_v_loss = 0.0
            total_pi_loss = 0.0
            count = 0

            for ctx_emb, doc_emb, scores in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}"
            ):
                # Device management
                ctx_emb = ctx_emb.to(self.device)
                doc_emb = doc_emb.to(self.device)
                scores = scores.to(self.device)

                # Forward pass
                outputs = model(ctx_emb, doc_emb)

                # Q-head loss
                q_loss = F.mse_loss(outputs["q_value"], scores)

                # V-head loss
                v_loss = self._expectile_loss(
                    scores - outputs["state_value"],
                    tau=self.cfg.get("expectile", 0.7),
                )

                # Policy head loss
                pi_loss = torch.tensor(0.0, device=self.device)
                if self.use_gild:
                    advantage = (
                        outputs["q_value"] - outputs["state_value"]
                    ).detach()
                    weights = torch.exp(self.beta * advantage)
                    weights = weights / weights.sum()

                    policy_probs = F.softmax(outputs["action_logits"], dim=-1)
                    entropy = -torch.sum(
                        policy_probs * torch.log(policy_probs + 1e-8), dim=-1
                    ).mean()

                    pi_loss = -(
                        F.log_softmax(outputs["action_logits"], dim=-1)
                        * weights
                    ).mean()
                    pi_loss += self.entropy_weight * entropy

                # Backward pass
                optimizers["q_head"].zero_grad()
                q_loss.backward()
                optimizers["q_head"].step()

                optimizers["v_head"].zero_grad()
                v_loss.backward()
                optimizers["v_head"].step()

                if self.use_gild:
                    optimizers["pi_head"].zero_grad()
                    pi_loss.backward()
                    optimizers["pi_head"].step()

                # Track losses
                total_q_loss += q_loss.item() * ctx_emb.size(0)
                total_v_loss += v_loss.item() * ctx_emb.size(0)
                total_pi_loss += pi_loss.item() * ctx_emb.size(0)
                count += ctx_emb.size(0)

            # End of epoch
            avg_q = total_q_loss / count
            avg_v = total_v_loss / count
            avg_pi = total_pi_loss / count

            # Early stopping
            if avg_q < best_loss - self.early_stopping_min_delta:
                best_loss = avg_q
                patience_counter = 0
                # Save best model
                torch.save(
                    model.encoder.state_dict(), f"{output_dir}/encoder.pt"
                )
                torch.save(
                    model.q_head.state_dict(), f"{output_dir}/q_head.pt"
                )
                torch.save(
                    model.v_head.state_dict(), f"{output_dir}/v_head.pt"
                )
                torch.save(
                    model.pi_head.state_dict(), f"{output_dir}/pi_head.pt"
                )
            else:
                patience_counter += 1

            # Log epoch
            self.logger.log(
                "SICQLTrainingEpoch",
                {
                    "epoch": epoch + 1,
                    "q_loss": avg_q,
                    "v_loss": avg_v,
                    "pi_loss": avg_pi,
                    "lr": optimizers["q_head"].param_groups[0]["lr"],
                },
            )

            # Check for early stopping
            if patience_counter >= self.early_stopping_patience:
                self.logger.log(
                    "SICQLEarlyStopping",
                    {"epoch": epoch + 1, "best_loss": best_loss},
                )
                break

        self.logger.log("SICQLTrainingComplete", {"best_loss": best_loss})
        return model

    def _save_model(self, model, dimension, stats):
        """Save SICQL model components"""
        locator = super().get_locator(dimension)
        # Save components separately
        torch.save(model.encoder.state_dict(), locator.encoder_file())
        torch.save(model.q_head.state_dict(), locator.q_head_file())
        torch.save(model.v_head.state_dict(), locator.v_head_file())
        torch.save(model.pi_head.state_dict(), locator.pi_head_file())

        # Calculate policy metrics
        policy_logits = model.pi_head.get_policy_weights().tolist()
        policy_probs_tensor = F.softmax(torch.tensor(policy_logits), dim=-1)
        policy_probs = policy_probs_tensor.tolist()
        policy_entropy = -torch.sum(
            policy_probs_tensor * torch.log(policy_probs_tensor + 1e-8)
        ).item()
        policy_stability = max(policy_probs)


        # Build metadata
        meta = {
            "dim": self.dim,
            "hdim": self.hdim,
            "dimension": dimension,
            "version": self.cfg.get("model_version", "v1"),
            "avg_q_loss": float(stats["avg_q_loss"]),
            "avg_v_loss": float(stats["avg_v_loss"]),
            "avg_pi_loss": float(stats["avg_pi_loss"]),
            "policy_entropy": float(policy_entropy),
            "policy_stability": float(policy_stability),
            "policy_logits": policy_logits,
            "policy_probs": policy_probs,
            "embedding_type": self.embedding_type,
            "max_value": 100,
            "min_value": 0,
            "device": str(self.device), 
            "timestamp": datetime.utcnow().isoformat(),
        }

        super()._save_meta_file(meta, dimension)
        return meta

    def run(self, context: dict) -> dict:
        """Main entry point for training"""
        documents = context.get("documents", [])
        # Train each dimension
        results = {}
        for dim in self.dimensions:
            # Get training samples
            samples = self._get_samples(context, documents, dim)
            if not samples:
                continue

            # Train model
            stats = self.train(samples, dim)
            if "error" in stats:
                continue

            # Update belief cartridges
            self._update_belief_cartridge(context, dim, stats)
            results[dim] = stats

        # Update context with results
        context["training_stats"] = results
        return context

    def _get_samples(self, context, documents, dim):
        """Get training samples for dimension"""
        samples = []
        goal = context.get("goal", {})
        for doc in documents:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            score = self.memory.scores.get_score(goal.id, scorable.id)
            if score:
                samples.append(
                    {
                        "title": goal.get("goal_text", ""),
                        "output": scorable.text,
                        "score": score.score,
                    }
                )
        return samples

    def _update_belief_cartridge(self, context, dim, stats):
        """Update belief cartridges with policy stats"""
        policy_logits = stats.get("policy_logits", [0.3, 0.7, 0.0])
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1).tolist()

        # Build belief cartridge
        cartridge = BeliefCartridgeORM(
            title=f"{dim} policy",
            content=f"Policy head weights: {policy_probs}",
            goal_id=context.get("goal_id"),
            domain=dim,
            policy_logits=policy_probs,
            policy_entropy=stats.get("policy_entropy", 1.05),
            policy_stability=stats.get("policy_stability", 0.82),
        )
        self.memory.session.add(cartridge)
        self.memory.session.commit()
