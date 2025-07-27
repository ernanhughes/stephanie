# stephanie/scoring/mrq/model.py
import torch
from torch import nn

from stephanie.scoring.model.policy_head import PolicyHead
from stephanie.scoring.model.q_head import QHead
from stephanie.scoring.model.v_head import VHead
from stephanie.scoring.model.text_encoder import TextEncoder


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