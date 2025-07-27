# stephanie/scoring/mrq/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, dim=4096, hdim=4096):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim * 2, hdim),  # Concatenate context + document
            nn.ReLU(),
            nn.Linear(hdim, dim),      # Keep the output same size
        )

    def forward(self, context_emb, doc_emb):
        concat = torch.cat([context_emb, doc_emb], dim=1)
        return self.encoder(concat)
