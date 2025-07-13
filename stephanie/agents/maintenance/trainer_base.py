# stephanie/agents/maintenance/trainer_base.py
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.maintenance.model_evolution_manager import \
    ModelEvolutionManager
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.preference_pair_builder import PreferencePairBuilder
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.utils.file_utils import save_json
from stephanie.utils.model_utils import get_model_path, save_model_with_version


class TrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = cfg.get("model_type", "ebt")
        self.output_key = cfg.get("output_key", "training_pairs")
        self.target_type = cfg.get("target_type", "document")
        self.encoder = TextEncoder().to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.value_predictor = ValuePredictor().to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.evolution_manager = ModelEvolutionManager(
            self.cfg, self.memory, self.logger
        )

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")

        from stephanie.scoring.mrq.preference_pair_builder import \
            PreferencePairBuilder

        # Build contrastive training pairs grouped by scoring dimension
        builder = PreferencePairBuilder(db=self.memory.session, logger=self.logger)
        training_pairs = builder.get_training_pairs_by_dimension(goal=goal_text)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train one model per scoring dimension (e.g. clarity, novelty, etc.)
        for dim, pairs in training_pairs.items():
            if not pairs:
                continue

            self.logger.log(
                "DocumentEBTTrainingStart", {"dimension": dim, "num_pairs": len(pairs)}
            )

            # Construct dataset and dataloader; normalize scores between 50–100
            ds = DocumentEBTDataset(pairs, min_score=50, max_score=100)
            dl = DataLoader(
                ds,
                batch_size=8,
                shuffle=True,
                collate_fn=lambda b: collate_ebt_batch(
                    b, self.memory.embedding, device
                ),
            )

            # Create model for this dimension
            model = EBTModel().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
            loss_fn = nn.MSELoss()

            # Training loop for fixed number of epochs
            for epoch in range(10):
                model.train()
                total_loss = 0.0
                for ctx_enc, cand_enc, labels in dl:
                    preds = model(ctx_enc, cand_enc)  # Predict score given (goal, doc)
                    loss = loss_fn(preds, labels)  # Compare against normalized label

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dl)
                self.logger.log(
                    "DocumentEBTEpoch",
                    {
                        "dimension": dim,
                        "epoch": epoch + 1,
                        "avg_loss": round(avg_loss, 5),
                    },
                )

            # Save trained model weights to disk
            model_path = f"{get_model_path(self.model_type, self.target_type, dim)}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(model.state_dict().keys())
            torch.save(model.state_dict(), model_path)
            self.logger.log(
                "DocumentEBTModelSaved", {"dimension": dim, "path": model_path}
            )

            # Save score normalization metadata for this dimension
            meta_path = model_path.replace(".pt", ".meta.json")
            normalization = ds.get_normalization()
            save_json(normalization, meta_path)

            self._save_and_promote_model(model, self.model_type, self.target_type, dim)

        context[self.output_key] = training_pairs
        return context

    def _save_and_promote_model(self, model, model_type, target_type, dimension):
        # Generate new version ID
        version = self._generate_version(model_type, target_type, dimension)

        # Save model with version
        version_path = save_model_with_version(
            model.state_dict(), model_type, target_type, dimension, version
        )

        # Log in DB
        model_id = self.evolution_manager.log_model_version(
            model_type=model_type,
            target_type=target_type,
            dimension=dimension,
            version=version,
            performance=self._get_validation_metrics(),  # e.g., accuracy, loss
        )

        # Get current best model
        current = self.evolution_manager.get_best_model(
            model_type, target_type, dimension
        )

        # Compare performance and promote if better
        if self.evolution_manager.check_model_performance(
            new_perf=self._get_validation_metrics(),
            old_perf=current["performance"] if current else {},
        ):
            self.evolution_manager.promote_model_version(model_id)
            self.logger.log(
                "ModelPromoted",
                {
                    "model_type": model_type,
                    "dimension": dimension,
                    "version": version,
                    "path": version_path,
                },
            )
        else:
            self.logger.log(
                "ModelNotPromoted",
                {
                    "model_type": model_type,
                    "dimension": dimension,
                    "new_version": version,
                    "current_version": current["version"] if current else None,
                },
            )


def collate_ebt_batch(batch, embedding_store, device):
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
