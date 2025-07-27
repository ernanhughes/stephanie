# stephanie/scoring/ebt/srft_refinement_trainer.py

import os
from datetime import datetime
from typing import Dict, List, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.ebt_mixin import EBTMixin
from stephanie.scoring.ebt.srft_refinement_dataset import SRFTRefinementDataset
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.utils.file_utils import save_json
from stephanie.utils.model_locator import ModelLocator


class SRFTRefinementTrainer(BaseAgent, EBTMixin):
    def __init__(self, cfg, memory=None, logger=None):
        BaseAgent.__init__(self, cfg, memory, logger)
        EBTMixin.__init__(self, cfg.get("ebt", {}))

        self.epochs = cfg.get("epochs", 5)
        self.batch_size = cfg.get("batch_size", 8)
        self.lr = cfg.get("learning_rate", 1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "ebt"
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")
        self.embedding_type = self.memory.embedding.type


    def train_srft_model(
        self,
        dimension: str,
        examples: List[Dict],
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ):
        """
        Train an EBT model using SRFT loss (SFT + RL + Entropy weighting)
        """
        model = EBTModel().to(self.device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Prepare dataset and dataloader
        dataset = SRFTRefinementDataset(examples, min_score, max_score)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda b: dataset.collate_fn(b, self.memory.embedding, self.device)
        )

        sft_weight = 0.7
        rl_weight = 0.3
        margin = self.cfg.get("loss_margin", 0.05)

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in dataloader:
                ctx, orig_doc, ref_doc, llm_score, mrq_reward, orig_energy, refined_energy = batch

                # Forward pass
                e_orig = model(ctx, orig_doc)
                e_ref = model(ctx, ref_doc)

                # SFT Loss: Predict lower energy for refined doc vs. original
                energy_diff = e_orig - e_ref
                sft_loss = torch.relu(margin - energy_diff).mean()

                # RL Loss: Learn to reduce energy if reward improved
                reward_diff = mrq_reward  # assume precomputed improvement
                rl_loss = ((e_ref - e_orig) - reward_diff).pow(2).mean()

                # Entropy-aware weighting (weight less if high orig_energy)
                uncertainty = orig_energy.detach().clamp(min=1e-5)
                weight = 1.0 / (1.0 + uncertainty)

                loss = weight * (sft_weight * sft_loss + rl_weight * rl_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            self.logger.log("SRFTRefinementEpoch", {
                "dimension": dimension,
                "epoch": epoch + 1,
                "avg_loss": total_loss / len(dataloader)
            })

        path = self._save_model(model, dimension)
        self.logger.log("SRFTRefinementComplete", {
            "dimension": dimension,
            "model_path": path
        })

    def _save_model(self, model, dimension: str) -> str:
        """Save model and metadata"""
        localer = ModelLocator(
            root_dir=self.model_path,
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dimension,
            version=self.model_version,
            embedding_type=self.embedding_type
        )
        localer.ensure_dirs()
        torch.save(model.state_dict(), localer.model_file())

        meta_path = localer.meta_file()
        meta = {
            "training_date": datetime.utcnow().isoformat(),
            "version": self.model_version,
        }
        save_json(meta, meta_path)

        return localer.model_file()

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        refinement_data = context.get("refinement_data", [])

        if not refinement_data:
            refinement_data = self._fetch_refinement_examples(goal_text)

        by_dim = {}
        for item in refinement_data:
            dim = item["dimension"]
            by_dim.setdefault(dim, []).append(item)

        for dim, examples in by_dim.items():
            self.train_srft_model(dim, examples)

        context["trained"] = list(by_dim.keys())
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
            "llm_score": r.llm_score,
            "mrq_reward": r.mrq_reward,
            "original_energy": r.original_energy,
            "refined_energy": r.refined_energy,
        } for r in results]
