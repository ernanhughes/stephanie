# stephanie/scoring/training/knowledge_trainer.py
from __future__ import annotations

import os
import argparse
import logging
import json
import torch
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from typing import List, Dict
import torch.nn.functional as F
from sqlalchemy.orm import sessionmaker

from stephanie.database import engine  # your DB engine
from stephanie.dataloaders.knowledge_pair_builder import KnowledgePairBuilder
from stephanie.scoring.model.knowledge import RewardHead, dpo_lite_loss
from stephanie.utils.embeddings import get_embedding

# Setup
SessionLocal = sessionmaker(bind=engine)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PairDataset(Dataset):
    def __init__(self, pairs: List[Dict]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "z_pos": torch.tensor(pair["pos_emb"], dtype=torch.float32),
            "z_neg": torch.tensor(pair["neg_emb"], dtype=torch.float32),
            "domain": pair["domain"],
            "goal_id": pair["goal_id"]
        }

def collate_fn(batch):
    z_pos = torch.stack([item["z_pos"] for item in batch])
    z_neg = torch.stack([item["z_neg"] for item in batch])
    # L2 normalize (double-check)
    z_pos = F.normalize(z_pos, p=2, dim=-1)
    z_neg = F.normalize(z_neg, p=2, dim=-1)
    return {
        "z_pos": z_pos,
        "z_neg": z_neg,
        "domains": [item["domain"] for item in batch],
        "goal_ids": [item["goal_id"] for item in batch]
    }

def evaluate_pairwise_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            z_pos = batch["z_pos"].to(device)
            z_neg = batch["z_neg"].to(device)
            s_pos = model(z_pos)
            s_neg = model(z_neg)
            correct += (s_pos > s_neg).sum().item()
            total += s_pos.size(0)
    return correct / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="models/mrq_dpo")
    parser.add_argument("--d_embedding", type=int, default=1024)  # adjust to your embedder
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--limit_pairs", type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build pairs
    db = SessionLocal()
    builder = KnowledgePairBuilder(db, embedder_fn=get_embedding)
    pairs = builder.build_pairs(limit=args.limit_pairs)

    if len(pairs) < 100:
        logger.error("Not enough pairs to train. Exiting.")
        return

    # Split train/val
    split = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    train_loader = DataLoader(PairDataset(train_pairs), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PairDataset(val_pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model & Optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RewardHead(args.d_embedding, hidden=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Train
    logger.info("Starting training...")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            z_pos = batch["z_pos"].to(device)
            z_neg = batch["z_neg"].to(device)

            s_pos = model(z_pos)
            s_neg = model(z_neg)
            loss = dpo_lite_loss(s_pos, s_neg, beta=args.beta, margin=args.margin)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        # Validate
        val_acc = evaluate_pairwise_accuracy(model, val_loader, device)
        avg_loss = total_loss / max(1, steps)
        logger.info(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.output_dir, "best_reward_head.pt")
            torch.save(model, model_path)
            logger.info(f"✅ Saved best model to {model_path}")

    # Final evaluation
    logger.info(f"🎉 Training complete. Best Val Acc: {best_val_acc:.4f}")

    # Save config
    config = {
        "d_embedding": args.d_embedding,
        "hidden_dim": args.hidden_dim,
        "beta": args.beta,
        "margin": args.margin,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "best_val_acc": best_val_acc,
        "trained_at": datetime.now().isoformat()
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()