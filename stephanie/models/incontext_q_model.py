import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from stephanie.scoring.model.text_encoder import TextEncoder


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class ExpectileHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class PolicyHead(nn.Module):
    def __init__(self, input_dim, hdim, action_dim=3):
        """
        Multi-layer policy head using a hidden layer

        Args:
            input_dim: Dimension of the zsa vector
            hdim: Hidden layer size
            action_dim: Number of discrete actions (e.g. MRQ, SVM, EBT)
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, action_dim)
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization"""
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.linear(x)  # raw logits; use softmax externally if needed

class InContextQModel(nn.Module):
    def __init__(self, dim, hdim, action_dim=3, device="cpu"):
        super().__init__()
        print(f"Initializing InContextQModel with dim={dim}, hdim={hdim}, action_dim={action_dim}, device={device}")
        self.device = device
        self.encoder = TextEncoder(dim, hdim).to(device)
        self.q_head = MLP(dim, hdim, 1).to(device)
        self.v_head = ExpectileHead(dim, hdim).to(device)
        self.pi_head = PolicyHead(dim, hdim, action_dim).to(device)

    def forward(self, prompt_emb, output_emb):
        prompt_emb = prompt_emb.to(self.device)
        output_emb = output_emb.to(self.device)

        # Ensure proper shape
        if prompt_emb.dim() == 1:
            prompt_emb = prompt_emb.unsqueeze(0)  # Add batch dim if missing
        if output_emb.dim() == 1:
            output_emb = output_emb.unsqueeze(0)

        zsa = self.encoder(prompt_emb, output_emb)  # [batch, hdim]
        
        # Get outputs without over-squeezing
        q_value = self.q_head(zsa)  # [batch, 1]
        state_value = self.v_head(zsa)  # [batch, 1]
        action_logits = self.pi_head(zsa)  # [batch, action_dim]
        
        # Ensure 2D structure for policy
        if action_logits.dim() == 1:
            action_logits = action_logits.unsqueeze(0)  # [1, action_dim]
        
        # For single-action spaces, flatten to [batch]
        if action_logits.size(1) == 1:
            action_logits = action_logits.squeeze(1)  # [batch]

        # Handle NaNs (critical for regression tuner)
        if torch.isnan(action_logits).any():
            self.logger.log("NaNPredicted", {
                "action_logits": action_logits.tolist()
            })
            action_logits = torch.zeros_like(action_logits)  # Fallback
        
        # Compute action probs (safe softmax)
        try:
            action_probs = F.softmax(action_logits, dim=-1)
        except Exception as e:
            self.logger.log("SoftmaxFailed", {
                "logits": action_logits.tolist(),
                "error": str(e)
            })
            action_probs = torch.ones_like(action_logits) / action_logits.size(-1)

        return {
            "zsa": zsa,
            "q_value": q_value,
            "state_value": state_value,
            "action_logits": action_logits,
            "action_probs": action_probs,
        }

    @classmethod
    def load_from_path(cls, model_path: str, dim_name: str, device="cpu"):
        """
        Load model weights from a directory:
        - {model_path}/{dim_name}_encoder.pt
        - {model_path}/{dim_name}_q.pt
        - {model_path}/{dim_name}_v.pt
        - {model_path}/{dim_name}_pi.pt
        """
        print(f"Loading InContextQModel from {model_path} for dimension {dim_name} on device {device}")
        encoder_path = os.path.join(model_path, f"{dim_name}_encoder.pt")
        q_path = os.path.join(model_path, f"{dim_name}_q.pt")
        v_path = os.path.join(model_path, f"{dim_name}_v.pt")
        pi_path = os.path.join(model_path, f"{dim_name}_pi.pt")

        with open(os.path.join(model_path, f"{dim_name}.meta.json")) as f:
            meta = json.load(f)

        dim = meta.get("dim", 1024)
        hdim = meta.get("hdim", 512)

        model = cls(dim=dim, hdim=hdim, device=device)
        print(f"Model initialized with dim={dim}, hdim={hdim}")
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        model.q_head.load_state_dict(torch.load(q_path, map_location=device))
        model.v_head.load_state_dict(torch.load(v_path, map_location=device))
        model.pi_head.load_state_dict(torch.load(pi_path, map_location=device))
        model.eval()
        return model
