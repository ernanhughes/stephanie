# stephanie/trainers/contrastive_knowledge_trainer.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class KnowledgePairDataset(Dataset):
    """
    Builds pairs of (turn_a, turn_b) where turn_a is more 'knowledgeable' than turn_b.
    """
    def __init__(self, cases, tokenizer, max_len=512):
        self.pairs = []
        for i in range(len(cases)):
            for j in range(len(cases)):
                if cases[i]["stars"] > cases[j]["stars"]:
                    self.pairs.append((cases[i], cases[j]))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        case_a, case_b = self.pairs[idx]
        a_enc = self.tokenizer(case_a["text"], truncation=True, max_length=self.max_len, return_tensors="pt")
        b_enc = self.tokenizer(case_b["text"], truncation=True, max_length=self.max_len, return_tensors="pt")
        return a_enc, b_enc, 1.0  # label always 1: a > b


class ContrastiveKnowledgeModel(nn.Module):
    def __init__(self, encoder, hidden_dim=256):
        super().__init__()
        self.encoder = encoder  # e.g., transformer encoder
        self.fc = nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.fc(pooled)


class ContrastiveTrainer:
    def __init__(self, model, lr=1e-5):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.MarginRankingLoss(margin=0.5)

    def train(self, dataloader, epochs=3, device="cuda"):
        self.model.to(device)
        for epoch in range(epochs):
            for batch in dataloader:
                (a_enc, b_enc, label) = batch
                a_enc = {k: v.squeeze(0).to(device) for k, v in a_enc.items()}
                b_enc = {k: v.squeeze(0).to(device) for k, v in b_enc.items()}
                label = torch.tensor([1.0], device=device)  # a > b

                score_a = self.model(**a_enc)
                score_b = self.model(**b_enc)

                loss = self.criterion(score_a, score_b, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch}: Loss {loss.item()}")
