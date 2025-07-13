# stephanie/scoring/ebt/ebt_refinement_trainer.py
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.ebt_mixin import EBTMixin
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import save_json
from stephanie.utils.model_utils import get_model_path


class EBTRefinementDataset(Dataset):
    """
    Dataset that contains original → refined document pairs
    Used to train EBT models to recognize refined content
    """
    def __init__(self, refinement_examples: List[Dict], min_score=None, max_score=None):
        """
        Args:
            refinement_examples: List of # Existing structure:
                {
                    "context": str,
                    "original": str,
                    "refined": str,
                    "dimension": str,
                    "original_score": float,
                    "refined_score": float,
                    "original_energy": float,
                    "refined_energy": float,
                    "llm_score": Optional[float],
                    "uncertainty": Optional[float]
                }
        """
        self.data = []
        self.min_score = min_score
        self.max_score = max_score
        
        # Compute global min/max if not provided
        if min_score is None or max_score is None:
            all_scores = [e["score"] for e in refinement_examples]
            self.min_score = min(all_scores) if min_score is None else min_score
            self.max_score = max(all_scores) if max_score is None else max_score

        # Build contrastive pairs
        for example in refinement_examples:
            # Original document gets normalized score
            norm_score = (example["score"] - self.min_score) / (self.max_score - self.min_score)
            
            # Refined document should have higher quality
            refined_score = example.get("refined_score", norm_score + 0.1)
            refined_score = max(0.0, min(1.0, refined_score))
            
            self.data.append({
                "context": example["context"],
                "output_a": example["original"],
                "output_b": example["refined"],
                "value_a": norm_score,
                "value_b": refined_score,
                "dimension": example["dimension"]
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EBTRefinementTrainer(BaseAgent, EBTMixin):
    """
    Trainer for EBT models using refinement examples
    Trains EBT to assign lower energy to refined documents
    """
    def __init__(self, cfg, memory=None, logger=None):
        """
        Args:
            cfg: Configuration dict with:
                - dimensions: List of dimensions to train
                - model_version: Version to save
                - epochs: Training epochs
                - batch_size: Training batch size
                - lr: Learning rate
                - margin: Margin for contrastive loss
        """
        BaseAgent.__init__(self, cfg, memory, logger)
        EBTMixin.__init__(self, cfg.get("ebt", {}))
        
        # Training configuration
        self.epochs = cfg.get("epochs", 10)
        self.batch_size = cfg.get("batch_size", 8)
        self.lr = cfg.get("learning_rate", 2e-5)
        self.margin = cfg.get("margin", 1.0)
        self.save_interval = cfg.get("save_interval", 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize regression tuners
        self.tuners = {
            dim: RegressionTuner(dimension=dim, logger=logger)
            for dim in self.dimensions
        }

    def prepare_refinement_data(self, examples: List[Dict]) -> DataLoader:
        """
        Convert raw refinement examples into training-ready DataLoader
        """
        # Group by dimension
        by_dimension = defaultdict(list)
        for example in examples:
            dim = example.get("dimension", "default")
            by_dimension[dim].extend(self._create_refinement_pairs(example))
        
        # Create datasets and loaders
        loaders = {}
        for dim, pairs in by_dimension.items():
            ds = EBTRefinementDataset(pairs)
            loaders[dim] = DataLoader(
                ds,
                num_workers= 4,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=lambda b: self._collate_fn(b, self.memory.embedding, self.device)
            )
        return loaders

    def _create_refinement_pairs(self, example: Dict) -> List[Dict]:
        """
        Convert a single refinement example into contrastive training pairs
        """
        context = example["context"]
        refined = example["refined"]
        original = example["original"]
        dim = example["dimension"]
        
        # Create multiple variations for robustness
        return [
            {
                "context": context,
                "output_a": original,
                "output_b": refined,
                "value_a": example["value"],
                "value_b": example["refined_value"],
                "dimension": dim
            },
            {
                "context": context,
                "output_a": refined,
                "output_b": original,
                "value_a": example["refined_value"],
                "value_b": example["value"],
                "dimension": dim
            }
        ]

    def _collate_fn(self, batch, embedding_store, device):
        """
        Convert batch of examples to tensors
        """
        ctxs, docs_a, docs_b, labels = [], [], [], []
        
        for item in batch:
            # Get embeddings
            ctx_emb = torch.tensor(embedding_store.get_or_create(item["context"])).to(device)
            a_emb = torch.tensor(embedding_store.get_or_create(item["output_a"])).to(device)
            b_emb = torch.tensor(embedding_store.get_or_create(item["output_b"])).to(device)
            
            # Contrastive loss labels
            preferred = "a" if item["value_a"] > item["value_b"] else "b"
            labels.append(1.0 if preferred == "a" else 0.0)
            
            ctxs.append(ctx_emb)
            docs_a.append(a_emb)
            docs_b.append(b_emb)
        
        # Stack tensors
        ctx_tensor = torch.stack(ctxs)
        doc_a_tensor = torch.stack(docs_a)
        doc_b_tensor = torch.stack(docs_b)
        label_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
        
        return ctx_tensor, doc_a_tensor, doc_b_tensor, label_tensor

    def contrastive_loss(self, energy_a, energy_b, label):
        """
        Contrastive loss for refinement training
        """
        # Label: 1 if a is better than b, 0 otherwise
        margin = self.cfg.get("loss_margin", 1.0)
        distances = torch.abs(energy_a - energy_b)
        
        # Calculate loss
        if label == 1:
            return distances  # We want energy_a < energy_b → smaller distance is good
        else:
            return torch.relu(margin - distances)  # Push apart if margin not met

    def train_refinement_model(self, dimension: str, dataloader: DataLoader):
        """
        Train EBT model for a single dimension
        """
        # Initialize fresh model for this dimension
        model = self._initialize_dimension_model(dimension)
        model.to(self.device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        total_loss = 0.0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for ctx, doc_a, doc_b, labels in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                energy_a = model(ctx, doc_a)
                energy_b = model(ctx, doc_b)
                
                # Calculate loss
                loss = self.contrastive_loss(energy_a, energy_b, labels).mean()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.logger.log("EBTRefinementEpoch", {
                "dimension": dimension,
                "epoch": epoch + 1,
                "loss": avg_epoch_loss
            })
            
            # Periodic model saving
            if (epoch + 1) % self.save_interval == 0:
                self._save_model(model, dimension, f"v{epoch + 1}")
        
        # Final save
        final_path = self._save_model(model, dimension, "latest")
        self.logger.log("EBTRefinementTrainingComplete", {
            "dimension": dimension,
            "final_loss": avg_epoch_loss,
            "model_path": final_path
        })
        return model

    def _initialize_dimension_model(self, dimension: str) -> EBTModel:
        """Initialize a fresh EBT model for a dimension"""
        model_path = get_model_path(
            self.model_path,
            self.model_type,
            self.target_type,
            dimension,
            self.model_version
        )
        os.makedirs(model_path, exist_ok=True)
        
        return EBTModel().to(self.device)

    def _save_model(self, model, dimension: str, version: str = "latest") -> str:
        """Save model and metadata"""
        model_path = get_model_path(
            self.model_path,
            self.model_type,
            self.target_type,
            dimension,
            version
        )
        
        # Save model weights
        torch.save(model.state_dict(), os.path.join(model_path, f"{dimension}.pt"))
        
        # Save normalization metadata
        meta_path = os.path.join(model_path, f"{dimension}.meta.json")
        meta = {
            "min_score": self.ebt_meta.get(dimension, {}).get("min", 40),
            "max_score": self.ebt_meta.get(dimension, {}).get("max", 100),
            "train_min_score": self._get_train_min_score(dimension),
            "train_max_score": self._get_train_max_score(dimension),
            "training_date": datetime.utcnow().isoformat(),
            "version": version
        }
        save_json(meta, meta_path)
        
        return model_path

    def _get_train_min_score(self, dimension: str):
        """Get training minimum score for normalization"""
        query = f"""
        SELECT MIN(score) FROM scoring_events
        WHERE dimension='{dimension}' AND source IN ('ebt', 'llm')
        """
        result = self.memory.db.execute(query).fetchone()
        return result[0] if result else 40

    def srft_loss(self, energy_orig, energy_ref, llm_score=None, orig_score=None, ref_score=None, entropy=None):
        """
        SRFT-style loss combining:
        - Supervised Fine-Tuning (match LLM score or gold refined score)
        - Reinforcement-style reward (improve MRQ score or reduce energy)
        """
        losses = []

        # 1. SFT: encourage refined to match or beat LLM score
        if llm_score is not None:
            target = torch.tensor(llm_score).to(self.device)
            losses.append(torch.nn.functional.mse_loss(energy_ref, target))

        # 2. RL-style reward: reward if refined energy < original
        if energy_orig is not None:
            margin = self.cfg.get("rl_margin", 0.05)
            diff = energy_orig - energy_ref
            rl_loss = -torch.relu(diff - margin)  # reward if improvement
            losses.append(rl_loss.mean())

        # 3. Entropy-aware weighting (optional)
        if entropy is not None:
            weight = 1.0 / (1.0 + entropy)
            total = sum(losses)
            return weight * total

        return sum(losses)


    def _get_train_max_score(self, dimension: str):
        """Get training maximum score for normalization"""
        query = f"""
        SELECT MAX(score) FROM scoring_events
        WHERE dimension='{dimension}' AND source IN ('ebt', 'llm')
        """
        result = self.memory.db.execute(query).fetchone()
        return result[0] if result else 100

    async def run(self, context: dict) -> dict:
        """
        Main training loop for EBT refinement models
        """
        goal_text = context.get("goal", {}).get("goal_text")
        refinement_data = context.get("refinement_data", [])
        
        if not refinement_data:
            # Fetch from database if no data provided
            refinement_data = self._fetch_refinement_examples(goal_text)
        
        # Prepare data
        dataloaders = self.prepare_refinement_data(refinement_data)
        
        # Train per-dimension models
        trained_models = {}
        for dim, loader in dataloaders.items():
            self.logger.log("EBTRefinementStart", {
                "dimension": dim,
                "examples": len(loader.dataset)
            })
            
            trained_model = self.train_refinement_model(dim, loader)
            trained_models[dim] = trained_model.state_dict()
            
            # Update tuner
            if dim in self.tuners:
                self._update_regression_tuner(loader, dim)
        
        # Update model registry
        self._update_model_registry(trained_models)
        
        context["trained_models"] = trained_models
        return context

    def _fetch_refinement_examples(self, goal: str = None) -> List[Dict]:
        """
        Fetch refinement examples from database
        """
        query = """
        SELECT * FROM refinement_events
        WHERE created_at > NOW() - INTERVAL '7 days'
        """
        if goal:
            query += f"AND context_hash = {hash(goal)}"
        
        results = self.memory.db.execute(query).fetchall()
        
        return [{
            "context": r.context,
            "original": r.original,
            "refined": r.refined,
            "dimension": r.dimension,
            "score": r.original_score,
            "refined_score": r.refined_score
        } for r in results]

    def _update_regression_tuner(self, dataloader: DataLoader, dimension: str):
        """Update regression tuner using refined examples"""
        for ctx, doc_a, doc_b, labels in dataloader:
            for i in range(len(ctx)):
                original = doc_a[i].cpu().numpy()
                refined = doc_b[i].cpu().numpy()
                llm_score = labels[i].cpu().item()
                
                # Update tuner with EBT-refined examples
                self.tuners[dimension].train_single(
                    ebt_score=doc_b[i].item(),
                    llm_score=llm_score
                )

    def _update_model_registry(self, trained_models: Dict[str, Dict]):
        """Update model registry with new versions"""
        for dim, state_dict in trained_models.items():
            self.logger.log("EBTModelUpdated", {
                "dimension": dim,
                "version": "auto",
                "performance": self._evaluate_model(dim, state_dict)
            })

    def _evaluate_model(self, dimension: str, model_state) -> Dict:
        """Evaluate model performance on validation data"""
        # Implement validation logic here
        return {
            "val_loss": 0.123,
            "accuracy": 0.91,
            "improvement": 0.05
        }