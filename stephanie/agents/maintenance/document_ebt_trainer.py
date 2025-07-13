# stephanie/agents/scoring/document_ebt_trainer.py
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.maintenance.model_evolution_manager import \
    ModelEvolutionManager
from stephanie.scoring.model.ebt_model import DocumentEBTScorer
from stephanie.scoring.mrq.preference_pair_builder import PreferencePairBuilder
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import save_json


class DocumentEBTDataset(Dataset):
    """Dataset that normalizes LLM scores for EBT training"""
    def __init__(self, contrast_pairs, min_score=None, max_score=None):
        self.data = []

        # Compute normalization range from data if not provided
        all_scores = []
        for pair in contrast_pairs:
            all_scores.extend([pair["value_a"], pair["value_b"]])
        self.min_score = min(all_scores) if min_score is None else min_score
        self.max_score = max(all_scores) if max_score is None else max_score

        # Convert to normalized training examples
        for pair in contrast_pairs:
            norm_a = (pair["value_a"] - self.min_score) / (self.max_score - self.min_score)
            norm_b = (pair["value_b"] - self.min_score) / (self.max_score - self.min_score)
            self.data.append((pair["title"], pair["output_a"], norm_a))
            self.data.append((pair["title"], pair["output_b"], norm_b))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def get_normalization(self):
        """Returns score range for inference-time rescaling"""
        return {"min": self.min_score, "max": self.max_score}


class DocumentEBTTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        """
        Unified EBT trainer with versioning and evolution management
        
        Args:
            cfg: Configuration dict with:
                - dimensions: List of scoring dimensions
                - model_version: Version to train
                - epochs: Training epochs
                - batch_size: Training batch size
                - lr: Learning rate
                - early_stopping: Enable early stopping
                - min_delta: Minimum improvement for early stopping
        """
        super().__init__(cfg, memory, logger)
        self.model_type = "ebt"
        self.target_type = "document"
        self.dimensions = cfg.get("dimensions", ["default"])
        self.model_version = cfg.get("model_version", "v1")
        self.epochs = cfg.get("epochs", 10)
        self.batch_size = cfg.get("batch_size", 8)
        self.lr = cfg.get("lr", 2e-5)
        self.early_stopping = cfg.get("early_stopping", True)
        self.min_delta = cfg.get("min_delta", 0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model evolution manager
        self.model_evolution = ModelEvolutionManager(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        
        # Build training data
        builder = PreferencePairBuilder(
            db=self.memory.session,
            logger=self.logger
        )
        training_pairs = builder.get_training_pairs_by_dimension(goal=goal_text)

        # Convert to flat list of contrast pairs
        all_contrast_pairs = []
        for dimension, pairs in training_pairs.items():
            for item in pairs:
                all_contrast_pairs.append({
                    "title": item["title"],
                    "output_a": item["output_a"],
                    "output_b": item["output_b"],
                    "value_a": item["value_a"],
                    "value_b": item["value_b"],
                    "dimension": dimension
                })

        if not all_contrast_pairs:
            self.logger.log("NoTrainingData", {"goal": goal_text})
            return context

        # Train per-dimension models
        trained_models = {}
        regression_tuners = {}

        for dim in self.dimensions:
            if dim not in training_pairs or not training_pairs[dim]:
                continue

            self.logger.log("DocumentEBTTrainingStart", {
                "dimension": dim, 
                "num_pairs": len(training_pairs[dim])
            })

            # Create dataset and dataloader
            ds = DocumentEBTDataset(training_pairs[dim], min_score=1, max_score=100)
            dl = DataLoader(
                ds,
                num_workers= 4,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=lambda b: self.collate_ebt_batch(b, self.memory.embedding, self.device)
            )

            # Initialize fresh model per dimension
            model = DocumentEBTScorer().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()

            best_loss = float('inf')
            epochs_no_improvement = 0
            
            # Training loop
            for epoch in range(self.epochs):
                model.train()
                total_loss = 0.0
                
                for ctx_enc, cand_enc, labels in dl:
                    preds = model(ctx_enc, cand_enc)
                    loss = loss_fn(preds, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()

                avg_loss = total_loss / len(dl)
                self.logger.log("DocumentEBTEpoch", {
                    "dimension": dim, 
                    "epoch": epoch + 1, 
                    "avg_loss": round(avg_loss, 5)
                })

                # Early stopping
                if self.early_stopping:
                    if avg_loss < best_loss - self.min_delta:
                        best_loss = avg_loss
                        epochs_no_improvement = 0
                        # Save best weights for this dimension
                        best_weights = model.state_dict().copy()
                    else:
                        epochs_no_improvement += 1
                        if epochs_no_improvement >= 3:
                            self.logger.log("EarlyStopping", {
                                "dimension": dim, 
                                "stopped_epoch": epoch + 1,
                                "best_loss": round(best_loss, 5)
                            })
                            model.load_state_dict(best_weights)
                            break

            # Save trained model
            dim_dir = os.path.join(
                self.cfg.get("model_save_path", "models"), 
                dim, 
                self.model_version
            )
            os.makedirs(dim_dir, exist_ok=True)
            
            # Save model weights
            model_path = os.path.join(dim_dir, "model.pt")
            torch.save(model.state_dict(), model_path)
            
            # Save normalization metadata
            meta_path = os.path.join(dim_dir, "meta.json")
            normalization = ds.get_normalization()
            save_json(normalization, meta_path)
            
            # Save tuner
            tuner = self._train_regression_tuner(dl)
            tuner_path = os.path.join(dim_dir, "tuner.json")
            tuner.save(tuner_path)
            
            # Register model version
            self.model_evolution.log_model_version(
                model_type=self.model_type,
                target_type=self.target_type,
                dimension=dim,
                version=self.model_version,
                performance={"loss": avg_loss}
            )
            
            trained_models[dim] = model.state_dict()
            regression_tuners[dim] = tuner

        # Update model registry
        self.model_evolution.promote_best_models(self.model_type, self.target_type)

        # Update context
        context[self.output_key] = {
            "trained_models": list(trained_models.keys()),
            "total_pairs": sum(len(p) for p in training_pairs.values())
        }
        return context

    def _train_regression_tuner(self, dataloader: DataLoader):
        """Train regression tuner for score calibration"""
        tuner = RegressionTuner(dimension="ebt", logger=self.logger)
        for ctx_enc, cand_enc, labels in dataloader:
            with torch.no_grad():
                predicted = self.model(ctx_enc, cand_enc).squeeze().cpu().numpy()
                actual = labels.cpu().numpy()
                for p, a in zip(predicted, actual):
                    tuner.train_single(p, a)
        return tuner
    
    def collate_ebt_batch(self, batch, embedding_store, device):
        # Custom batch collation for EBT dataset: fetch embeddings for goal and doc
        ctxs, docs, targets = zip(*batch)

        # Look up or create embeddings for each goal and candidate doc
        ctx_embs = [torch.tensor(embedding_store.get_or_create(c)).to(device) for c in ctxs]
        doc_embs = [torch.tensor(embedding_store.get_or_create(d)).to(device) for d in docs]
        labels = torch.tensor(targets, dtype=torch.float32).to(device)

        # Stack them into batched tensors for training
        ctx_tensor = torch.stack(ctx_embs)
        doc_tensor = torch.stack(doc_embs)

        return ctx_tensor, doc_tensor, labels
