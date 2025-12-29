# stephanie/scoring/model/in_context_q.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU

from stephanie.scoring.model.text_encoder import TextEncoder


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

class QHead(nn.Module):
    def __init__(self, zsa_dim, hdim):
        """
        Q-value estimator: Q(s,a) = E[reward | state, action]
        
        Args:
            zsa_dim: Dimension of encoded state-action vector
            hdim: Hidden layer dimension
        """
        super().__init__()
        self.model = nn.Sequential(
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
        Predict Q-value for (state, action) pair
        Args:
            zsa: Encoded state-action vector
        Returns:
            Q-value (scalar)
        """
        return self.model(zsa).squeeze()

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

class InContextQModel(nn.Module):
    def __init__(
        self, 
        encoder: TextEncoder,
        q_head: QHead,
        v_head: VHead,
        pi_head: PolicyHead,
        embedding_store,
        device="cpu"
    ):
        super().__init__()
        self.encoder = encoder.to(device)
        self.q_head = q_head.to(device)
        self.v_head = v_head.to(device)
        self.pi_head = pi_head.to(device)
        self.device = device
        self.embedding_store = embedding_store
    
    def forward(self, context_emb, doc_emb):
        """
        Forward pass through all heads
        
        Args:
            context_emb: Goal/prompt embedding
            doc_emb: Document/output embedding
        Returns:
            Dict containing Q-value, state value, and policy logits
        """
        # Ensure device alignment
        context_emb = context_emb.to(self.device)
        doc_emb = doc_emb.to(self.device)
        
        # Combine embeddings
        zsa = self.encoder(context_emb, doc_emb)
        
        # Forward through heads
        q_value = self.q_head(zsa)
        state_value = self.v_head(zsa)
        action_logits = self.pi_head(zsa)
        
        # Calculate advantage
        advantage = (q_value - state_value).detach()
        
        return {
            "q_value": q_value,
            "state_value": state_value,
            "action_logits": action_logits,
            "advantage": advantage
        }