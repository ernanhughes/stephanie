# stephanie/components/gap/scm/taps.py
from __future__ import annotations

from typing import Dict, Any
import torch

from typing import Dict, Any
import torch

class TinyTap:
    def __call__(self, internals: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "final_logits":    internals.get("final_logits"),      # [T,V] tensor
            "token_entropies": internals.get("token_entropies"),   # [T] tensor
            "latent":          internals.get("latent_trace"),      # [L,D] or [D]
            "attn_stats":      internals.get("attn_stats", {}),    # dict of floats
            "uncertainty01":   internals.get("entropy01", 0.0),
            "consistency01":   internals.get("consistency01", None),
            "ood_hat01":       internals.get("ood_hat01", None),
            "len_effect":      internals.get("len_effect", 0.0),
            "temp":            internals.get("temp01", 0.0),
            "agree_hat":       internals.get("agree01", 0.5),
            "steps":           internals.get("steps", None),
        }

class HRMTap:
    def __call__(self, internals: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "final_logits":    internals.get("final_logits"),
            "token_entropies": internals.get("token_entropies"),
            "latent":          internals.get("z_trace") or internals.get("latent"),
            "attn_stats":      internals.get("attn_stats", {}),
            "uncertainty01":   internals.get("energy01", 0.0),
            "consistency01":   internals.get("consistency01", None),
            "ood_hat01":       internals.get("ood_hat01", None),
            "len_effect":      internals.get("len_effect", 0.0),
            "temp":            internals.get("temp01", 0.0),
            "agree_hat":       internals.get("agree_hat", 0.5),
            "steps":           internals.get("steps", None),
        }
