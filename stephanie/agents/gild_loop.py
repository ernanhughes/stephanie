
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.gild_selector import GILDSelector
from stephanie.memory.scoring_store import ScoringStore
from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.model_locator import ModelLocator


class GILDQMAXTrainer(BaseAgent):
    def __init__(self, cfg, memory, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", ["alignment", "clarity", "novelty"])
        self.use_gild = cfg.get("use_gild", True)
        self.use_qmax = cfg.get("use_qmax", True)
        self.model_store = ScoringStore(memory.session, logger)
        self.tuners = {dim: RegressionTuner(dim=dim) for dim in self.dimensions}
        self.gild_selector = GILDSelector(memory.session, logger)
        self.uncertainty_threshold = cfg.get("uncertainty_threshold", 0.3)
        self.min_samples = cfg.get("min_samples", 100)
        self.model_locator = ModelLocator(cfg.get("model_path", "models"))
        
        # Track policy improvements
        self.policy_stats = {
            dim: {"drift": [], "entropy": [], "stability": []} 
            for dim in self.dimensions
        }

    async def run(self, context: dict) -> dict:
        self.train(context, context.get("documents", []))

    def train(self, context: dict, documents: list[dict]):
        """
        Main training loop that integrates:
        1. GILD for policy improvement
        2. Q-MAX for stable updates
        3. Epistemic monitoring
        """
        goal = context.get("goal", {})
        self.logger.log("TrainingStarted", {
            "goal": goal.get("goal_text", "")[:50] + "...",
            "document_count": len(documents),
            "dimensions": self.dimensions,
            "use_gild": self.use_gild,
            "use_qmax": self.use_qmax
        })
        
        # Build scorer models
        self.models = self._build_dimension_models()
        
        # Get expert demonstrations
        expert_demos = self._get_expert_demos(goal, documents)
        
        # Create training data
        train_data = self._prepare_training_data(goal, documents, expert_demos)
        
        # Train each dimension
        for dim in self.dimensions:
            self.logger.log("DimensionTrainingStarted", {"dimension": dim})
            dim_data = train_data[dim]
            
            if len(dim_data) < self.min_samples:
                self.logger.log("InsufficientSamples", {
                    "dimension": dim,
                    "sample_count": len(dim_data),
                    "threshold": self.min_samples
                })
                continue
                
            # Train model
            trained_model, stats = self._train_dimension(dim, dim_data)
            
            # Save model and stats
            self._save_model(context, trained_model, dim, stats)
            
            # Update policy stats
            self.policy_stats[dim].update(stats)
        
        self.logger.log("TrainingCompleted", {
            "policy_stats": self.policy_stats
        })
        return self.policy_stats

    def _build_dimension_models(self):
        """Build or load models for each dimension"""
        models = {}
        for dim in self.dimensions:
            model = self.model_locator.load_model(
                embedding_type=self.cfg.get("embedding_type", "hnet"),
                model_type="ebt",
                target_type="document",
                dimension=dim
            )
            
            if not model:
                model = EBTModel(
                    embedding_dim=self.cfg.get("embedding_dim", 1024),
                    hidden_dim=self.cfg.get("hidden_dim", 256),
                    num_actions=self.cfg.get("num_actions", 3),
                    device=self.device
                ).to(self.device)
            
            models[dim] = model
        return models

    def _get_expert_demos(self, goal, documents):
        """Get expert demonstrations from LLM or historical data"""
        expert_demos = {}
        for doc in documents:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            demo = self.gild_selector.select_best_scorer(goal, scorable)
            expert_demos[scorable.id] = demo
        return expert_demos

    def _prepare_training_data(self, goal, documents, expert_demos):
        """Prepare training data with expert demonstrations"""
        train_data = {dim: [] for dim in self.dimensions}
        
        for doc in documents:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            context_emb = torch.tensor(self.memory.embedding.get_or_create(goal["goal_text"]))
            doc_emb = torch.tensor(self.memory.embedding.get_or_create(scorable.text))
            
            for dim in self.dimensions:
                # Get expert demonstration
                expert = expert_demos.get(scorable.id, {}).get(dim, {})
                
                train_data[dim].append({
                    "context_emb": context_emb,
                    "doc_emb": doc_emb,
                    "llm_score": expert.get("llm_score", 0.5),
                    "expert_policy": expert.get("policy", [0.3, 0.7, 0.0]),
                    "dimension": dim
                })
        
        return train_data

    def _train_dimension(self, dim, data):
        """Train model for a single dimension with GILD+Q-MAX"""
        model = self.models[dim].to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.get("lr", 2e-4))
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=0.5, 
            patience=2
        )
        
        # Split data
        train_loader = self._create_dataloader(data)
        
        # Track training stats
        stats = {
            "q_losses": [],
            "v_losses": [],
            "pi_losses": [],
            "total_losses": [],
            "policy_entropies": []
        }
        
        # Training loop
        for epoch in range(self.cfg.get("epochs", 10)):
            model.train()
            epoch_q_loss = 0.0
            epoch_v_loss = 0.0
            epoch_pi_loss = 0.0
            
            for batch in train_loader:
                # Move to device
                ctx_emb = batch["context_emb"].to(self.device)
                doc_emb = batch["doc_emb"].to(self.device)
                llm_score = batch["llm_score"].to(self.device)
                expert_policy = torch.tensor(
                    batch["expert_policy"]
                ).to(self.device)
                
                # Forward pass
                outputs = model(ctx_emb, doc_emb)
                
                # Q-head loss
                q_loss = torch.nn.MSELoss()(
                    outputs["q_value"], 
                    llm_score
                )
                
                # V-head loss with expectile regression
                v_loss = self._expectile_loss(
                    outputs["q_value"], 
                    outputs["state_value"]
                )
                
                # Policy head loss (GILD)
                if self.use_gild:
                    pi_loss = self._gild_policy_loss(
                        outputs["action_logits"], 
                        outputs["advantage"]
                    )
                else:
                    pi_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                total_loss = (
                    q_loss * self.cfg.get("q_weight", 1.0) +
                    v_loss * self.cfg.get("v_weight", 0.5) +
                    pi_loss * self.cfg.get("pi_weight", 0.3)
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Track losses
                epoch_q_loss += q_loss.item()
                epoch_v_loss += v_loss.item()
                epoch_pi_loss += pi_loss.item()
            
            # End of epoch
            avg_q_loss = epoch_q_loss / len(train_loader)
            avg_v_loss = epoch_v_loss / len(train_loader)
            avg_pi_loss = epoch_pi_loss / len(train_loader)
            
            stats["q_losses"].append(avg_q_loss)
            stats["v_losses"].append(avg_v_loss)
            stats["pi_losses"].append(avg_pi_loss)
            
            # Update learning rate
            if self.use_qmax:
                scheduler.step(avg_q_loss)
            
            # Log epoch stats
            self.logger.log("TrainingEpoch", {
                "dimension": dim,
                "epoch": epoch + 1,
                "q_loss": avg_q_loss,
                "v_loss": avg_v_loss,
                "pi_loss": avg_pi_loss
            })
        
        # Final model stats
        return model, {
            "q_loss": stats["q_losses"][-1],
            "v_loss": stats["v_losses"][-1],
            "pi_loss": stats["pi_losses"][-1],
            "avg_q_loss": np.mean(stats["q_losses"]),
            "avg_v_loss": np.mean(stats["v_losses"]),
            "avg_pi_loss": np.mean(stats["pi_losses"]),
            "policy_entropy": self._calculate_policy_entropy(model, data),
            "policy_stability": self._calculate_policy_stability(model, data)
        }

    def _create_dataloader(self, data):
        """Create DataLoader for training"""
        # Convert to tensors
        context_embs = torch.stack([d["context_emb"] for d in data])
        doc_embs = torch.stack([d["doc_emb"] for d in data])
        llm_scores = torch.tensor([d["llm_score"] for d in data])
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            context_embs, doc_embs, llm_scores
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.get("batch_size", 8),
            shuffle=True
        )

    def _expectile_loss(self, q_values, v_values):
        """Compute expectile loss for V-head training"""
        diff = q_values - v_values
        expectile = self.cfg.get("expectile", 0.7)
        
        return torch.where(
            diff > 0,
            expectile * (diff ** 2),
            (1 - expectile) * (diff ** 2)
        ).mean()

    def _gild_policy_loss(self, action_logits, advantage):
        """GILD-style policy improvement loss"""
        if not self.use_gild:
            return torch.tensor(0.0, device=self.device)
        
        # Advantage-weighted regression
        weights = torch.exp(self.cfg.get("beta", 1.0) * advantage.detach())
        weights = weights / weights.sum()  # Normalize
        
        # Policy entropy regularization
        action_probs = F.softmax(action_logits, dim=-1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        
        # Policy loss with entropy regularization
        policy_loss = -(F.log_softmax(action_logits, dim=-1) * weights).mean()
        entropy_bonus = -self.cfg.get("entropy_weight", 0.01) * entropy.mean()
        
        return policy_loss + entropy_bonus

    def _calculate_policy_entropy(self, model, data):
        """Calculate policy entropy for a model"""
        with torch.no_grad():
            context_embs = torch.stack([d["context_emb"] for d in data])
            doc_embs = torch.stack([d["doc_emb"] for d in data])
            
            outputs = model(context_embs.to(self.device), doc_embs.to(self.device))
            action_probs = F.softmax(outputs["action_logits"], dim=-1)
            entropy = -torch.sum(
                action_probs * torch.log(action_probs + 1e-8), 
                dim=-1
            ).mean().item()
        
        return entropy

    def _calculate_policy_stability(self, model, data):
        """Calculate policy stability over time"""
        with torch.no_grad():
            context_embs = torch.stack([d["context_emb"] for d in data])
            doc_embs = torch.stack([d["doc_emb"] for d in data])
            
            # Get policy outputs
            outputs = model(context_embs.to(self.device), doc_embs.to(self.device))
            action_probs = F.softmax(outputs["action_logits"], dim=-1)
            actions = torch.argmax(action_probs, dim=-1)
            
            # Calculate action consistency
            unique_actions, counts = torch.unique(actions, return_counts=True)
            policy_stability = counts.max() / len(actions)
        
        return float(policy_stability)

    def _save_model(self, context, model, dimension, stats):
        """Save model and update belief cartridges"""
        # Save model weights
        self.model_locator.save_model(model, dimension)
        
        # Update belief cartridges with new policy
        if self.use_gild:
            self._update_belief_cartridges(context, model, dimension, stats)

    def _update_belief_cartridges(self, context, model, dimension, stats):
        """Update belief cartridges with new policy"""
        # Get top policy outputs
        policy_logits = model.pi_head.weight.data.mean(dim=0)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Create belief cartridge
        cartridge = BeliefCartridgeORM(
            title=f"{dimension} policy",
            content=f"Policy head weights: {policy_probs.tolist()}",
            goal_id=context.get("goal_id"),
            domain=dimension,
            policy_logits=policy_probs.tolist(),
            policy_entropy=stats["policy_entropy"],
            policy_stability=stats["policy_stability"]
        )
        
        self.memory.session.add(cartridge)
        self.memory.session.commit()

    def _log_policy_improvement(self, dimension, old_policy, new_policy):
        """Log policy improvement to database"""
        policy_diff = {
            f"action_{i}": new_policy[i] - old_policy[i]
            for i in range(len(new_policy))
        }
        
        self.logger.log("PolicyImprovement", {
            "dimension": dimension,
            "old_policy": old_policy,
            "new_policy": new_policy,
            "policy_diff": policy_diff
        })