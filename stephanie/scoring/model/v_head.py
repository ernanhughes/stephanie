# stephanie/scoring/model/v_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU


class VHead(nn.Module):
    def __init__(self, zsa_dim, hdim):
        """
        State value estimator using expectile regression
        
        Args:
            zsa_dim: Dimension of encoded state-action vector
            hdim: Hidden layer dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            Linear(zsa_dim, hdim),
            ReLU(),
            Linear(hdim, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, zsa):
        """
        Predict state value V(s)
        Args:
            zsa: Encoded state-action vector
        Returns:
            State value (scalar)
        """
        return self.net(zsa).squeeze()