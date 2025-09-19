# stephanie/agents/thought/sot_v01_collator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class SoTDataCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Each item in batch should have fields pre-assembled:
        #   prompt_text, target_text, move_label (int)
        prompts = [b["prompt_text"] for b in batch]
        targets = [b["target_text"] + self.tokenizer.eos_token for b in batch]

        # Tokenize separately to compute prompt lengths
        tok_prompt = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        tok_full   = self.tokenizer([p+t for p,t in zip(prompts, targets)],
                                    padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        input_ids = tok_full.input_ids
        attention_mask = tok_full.attention_mask

        labels = input_ids.clone()
        # mask prompt part
        for i, plen in enumerate(tok_prompt.input_ids.shape[1] for _ in prompts):
            # recompute exact prompt lengths per sample from non-pad count
            plen = int((tok_prompt.attention_mask[i] == 1).sum().item())
            labels[i, :plen] = -100

        prompt_lengths = []
        for i in range(len(prompts)):
            plen = int((tok_prompt.attention_mask[i] == 1).sum().item())
            prompt_lengths.append(plen)
        prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long)

        move_labels = torch.tensor([b["move_label"] for b in batch], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "move_labels": move_labels,
            "prompt_lengths": prompt_lengths
        }
