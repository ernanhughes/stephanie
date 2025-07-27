# stephanie/scoring/model/policy_head.py
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU


class PolicyHead(nn.Module):
    def __init__(self, zsa_dim, hdim, num_actions=3):
        super().__init__()
        self.linear = nn.Sequential(
            Linear(zsa_dim, hdim),
            ReLU(),
            Linear(hdim, num_actions)
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, zsa):
        return self.linear(zsa)

    def get_policy_weights(self):
        """
        Get the averaged weights of the final linear layer for policy logits.
        """
        final_linear_layer = self.linear[-1]
        return final_linear_layer.weight.data.mean(dim=0)
