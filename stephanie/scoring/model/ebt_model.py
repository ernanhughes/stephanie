# stephanie/scoring/model/ebt_model.py
import torch
from torch import nn
from torch.nn import functional as F


class EBTModel(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=256, num_actions=3, device="cpu"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.device = device

        # Encoder with attention
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Q head with learnable scaling
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # V head with expectile regression
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Policy head with policy entropy
        self.pi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # Learnable scaling factor
        self.scale_factor = nn.Parameter(torch.tensor(10.0))

    def forward(self, context_emb, output_emb):
        # Ensure device alignment
        context_emb = context_emb.to(self.device)
        output_emb = output_emb.to(self.device)
        
        # Combine embeddings
        combined = torch.cat([context_emb, output_emb], dim=-1)
        zsa = self.encoder(combined)
        
        # Q/V heads
        q_value = self.q_head(zsa).squeeze()
        state_value = self.v_head(zsa).squeeze()
        
        # Policy head
        action_logits = self.pi_head(zsa)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Compute advantage
        advantage = q_value - state_value
        
        # Scale final score
        final_score = q_value * torch.sigmoid(self.scale_factor).item()
        
        return {
            "q_value": q_value,
            "state_value": state_value,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "advantage": advantage,
            "score": final_score
        }
    