import torch
import torch.nn as nn


class PreferenceRanker(nn.Module):
    """Siamese network for pairwise preference ranking"""
    def __init__(self, embedding_dim=768, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, emb_a, emb_b):
        feat_a = self.encoder(emb_a)
        feat_b = self.encoder(emb_b)
        combined = torch.cat([feat_a, feat_b], dim=1)
        return self.comparator(combined).squeeze(1)


