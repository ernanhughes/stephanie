# stephanie/scoring/mrq/reward_based_trainer.py
import torch
import torch.nn as nn


class RewardBasedTrainer:
    def __init__(self, encoder, predictor, optimizer, device="cpu"):
        self.encoder = encoder
        self.predictor = predictor
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()
        self.device = device

    def update(self, context_emb, doc_emb, reward):
        self.encoder.eval()
        self.predictor.train()

        with torch.no_grad():
            z = self.encoder(context_emb, doc_emb)

        pred = self.predictor(z)
        target = torch.tensor([reward], device=self.device)
        loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
