# stephanie/agents/thought/sot_v01_multitask.py
from __future__ import annotations

import torch
import torch.nn as nn

MOVE_LABELS = ["VOICE","OUTLINE","CRITIQUE","CODE","REFACTOR","PLAN","MATH","DERIVE","SEARCH","RETRIEVE"]
MOVE2ID = {m:i for i,m in enumerate(MOVE_LABELS)}

class SoTMultiTaskWrapper(nn.Module):
    """
    Wraps a Causal LM with a small classifier head for move prediction.
    - lm: AutoModelForCausalLM
    """
    def __init__(self, lm, hidden_size:int, num_moves:int=len(MOVE_LABELS), move_loss_weight:float=0.2):
        super().__init__()
        self.lm = lm
        self.move_head = nn.Linear(hidden_size, num_moves)
        self.move_loss_weight = move_loss_weight

    def forward(self, input_ids=None, attention_mask=None, labels=None, move_labels=None, prompt_lengths=None):
        # LM forward
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lm_loss = out.loss

        # Move classification: take the hidden state at the last prompt token per sample
        # prompt_lengths: (B,) tensor with prompt length in tokens for each sample
        hidden_states = out.hidden_states[-1]     # (B, T, H)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        idxs = torch.clamp(prompt_lengths-1, min=0)
        reps = hidden_states[batch_idx, idxs]     # (B, H)

        logits = self.move_head(reps)             # (B, num_moves)
        move_loss = nn.functional.cross_entropy(logits, move_labels) if move_labels is not None else 0.0

        loss = lm_loss + self.move_loss_weight * move_loss
        return type("Out", (), {"loss": loss, "lm_loss": lm_loss, "move_loss": move_loss, "logits": out.logits, "move_logits": logits})
