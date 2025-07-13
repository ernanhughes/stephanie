# stephanie/scoring/model/ebt_model.py
import torch
from torch import nn


class EBTModel(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        # A small feedforward head that maps concatenated (goal + doc) embeddings to a single score
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),  # Input: goal + doc embeddings
            nn.ReLU(),
            nn.Linear(256, 1),  # Output: scalar score (before scaling)
        )
        # Learnable scaling factor to adjust output magnitude during training
        self.scale_factor = nn.Parameter(torch.tensor(10.0))

    def forward(self, ctx_emb, doc_emb):
        # Concatenate context (goal) and document embeddings
        combined = torch.cat([ctx_emb, doc_emb], dim=-1)
        # Run through MLP head and apply learnable scaling
        raw = self.head(combined).squeeze(-1)
        return raw * self.scale_factor
