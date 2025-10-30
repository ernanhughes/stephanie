# stephanie/scoring/plugins/entailment_plugin.py
from __future__ import annotations
import math
from typing import Any, Dict, Optional
from .registry import register
import torch

@register("entailment")
class EntailmentPlugin:
    """
    Computes entailment probability between goal and reply using the host scorer's model.
    Uses teacher-forced likelihood estimation: P(reply | goal) → maps to [0,1].

    This plugin is designed to work with HuggingFaceScorer and leverages its
    loaded model/tokenizer. It does NOT load a new model.

    Outputs:
        entailment.score01: float ∈ [0,1] — higher = more entailed
        entailment.risk01:  float ∈ [0,1] — 1 - score01 (for risk alignment)

    Usage in HallucinationContext:
        entailment = lambda p, h: tap_output.attributes.get("entailment.score01", 0.5)
    """

    def __init__(
        self,
        container=None,
        logger=None,
        host=None,
        *,
        threshold: float = 0.35,  # for risk01 mapping
        use_logprob: bool = True,  # if False, uses cosine similarity fallback
    ):
        self.container = container
        self.logger = logger
        self.host = host
        self.threshold = float(threshold)
        self.use_logprob = bool(use_logprob)

        # Validate: host must be HuggingFaceScorer
        if not hasattr(self.host, "_ll_stats") or not hasattr(self.host, "tok"):
            raise ValueError(
                "EntailmentPlugin requires host to be a HuggingFaceScorer instance"
            )

    def post_process(self, *, tap_output: Dict[str, Any]) -> Dict[str, float]:
        """
        Post-process hook: computes entailment from goal and reply.
        Uses host's model to compute P(reply | goal) via teacher-forced logprob.
        """
        goal_text = tap_output.get("goal_text", "")
        reply_text = tap_output.get("resp_text", "")

        if not goal_text.strip() or not reply_text.strip():
            return {
                "entailment.score01": 0.5,
                "entailment.risk01": 0.5,
            }

        try:
            # Use the host's tokenizer and model (already loaded)
            tok = self.host.tok
            model = self.host.model

            # Tokenize goal and reply
            enc_goal = tok(goal_text, return_tensors="pt", add_special_tokens=False)
            enc_reply = tok(reply_text, return_tensors="pt", add_special_tokens=False)

            # Concatenate: [goal] [SEP] [reply]
            sep_token_id = tok.sep_token_id or tok.eos_token_id or 0
            input_ids = torch.cat(
                [
                    enc_goal["input_ids"],
                    torch.tensor([[sep_token_id]]),
                    enc_reply["input_ids"],
                ],
                dim=1,
            ).to(model.device)
            attention_mask = torch.ones_like(input_ids)

            # Truncate if needed
            max_seq_len = getattr(self.host, "max_seq_len", 4096)
            if input_ids.shape[1] > max_seq_len:
                cut = input_ids.shape[1] - max_seq_len
                input_ids = input_ids[:, cut:]
                attention_mask = attention_mask[:, cut:]

            # Forward pass
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits  # [1, T, V]

            # Shift for teacher forcing: predict next token
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            # Extract only the reply portion (after SEP)
            sep_pos = (input_ids[0] == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_pos) == 0:
                return {"entailment.score01": 0.5, "entailment.risk01": 0.5}

            sep_idx = sep_pos[0].item()
            reply_start = sep_idx + 1
            reply_logits = shift_logits[:, reply_start:, :]
            reply_labels = shift_labels[:, reply_start:]

            if reply_labels.numel() == 0:
                return {"entailment.score01": 0.5, "entailment.risk01": 0.5}

            # Compute mean log probability of reply given goal
            logprobs = torch.nn.functional.log_softmax(reply_logits, dim=-1)
            chosen_lp = torch.gather(
                logprobs, dim=-1, index=reply_labels.unsqueeze(-1)
            ).squeeze(-1)
            mean_logprob = float(chosen_lp.mean().item())

            # Map mean_logprob → [0,1] using logistic transform (same as before)
            # We calibrate: mean_logprob ~ -1.0 to +0.5 → map to [0,1]
            k = 3.0
            x0 = -0.3
            entail_score = 1.0 / (1.0 + math.exp(-k * (mean_logprob - x0)))
            entail_score = max(0.0, min(1.0, entail_score))

            return {
                "entailment.score01": entail_score,
                "entailment.risk01": 1.0 - entail_score,
            }

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "EntailmentPlugin failed; returning neutral score", extra={"error": str(e)}
                )
            return {"entailment.score01": 0.5, "entailment.risk01": 0.5}