import torch
import torch.nn as nn
import torch.nn.functional as F


# TextEncoder for embedding prompts and hypotheses
class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=1024, zs_dim=512, za_dim=256, zsa_dim=512, hdim=1024):
        super().__init__()
        self.zs_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, zs_dim)
        )
        self.za_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, za_dim)
        )
        self.zsa_mlp = nn.Sequential(
            nn.Linear(zs_dim + za_dim, zsa_dim),
            nn.ReLU(),
            nn.Linear(zsa_dim, zsa_dim)
        )

    def forward(self, prompt_emb, response_emb):
        zs = F.relu(self.zs_mlp(prompt_emb))
        za = F.relu(self.za_mlp(response_emb))
        zsa = self.zsa_mlp(torch.cat([zs, za], dim=1))
        return zsa
