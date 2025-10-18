# stephanie/components/gap/adapters/hf_vpm_adapter.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFVPMAdapter:
    """
    HuggingFace generator that returns a VPM-like trace:
      - generated text
      - ids, tokens
      - per-step chosen logprob and top-k alternatives
      - raw logits (optional)
    You can pass this trace into the same SCM computation used for Tiny/HRM.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        model_name = self.cfg.get("model_name", "gpt2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(self.cfg.get("max_new_tokens", 128))
        self.top_k = int(self.cfg.get("top_k", 20))
        self.top_p = float(self.cfg.get("top_p", 0.95))
        self.temperature = float(self.cfg.get("temperature", 0.8))
        self.do_sample = bool(self.cfg.get("do_sample", True))
        self.return_logits = bool(self.cfg.get("return_logits", False))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # ensure we can decode step-by-step
        if self.tokenizer.pad_token is None:
            # make pad = eos for causal models
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.cfg.get("fp16", True) and torch.cuda.is_available() else None,
        ).to(self.device)
        self.model.eval()

        if self.logger:
            self.logger.log("HFVPMAdapterInit", {
                "model": model_name, "device": str(self.device),
                "max_new_tokens": self.max_new_tokens,
                "top_k": self.top_k, "top_p": self.top_p,
                "temperature": self.temperature, "do_sample": self.do_sample
            })

    @torch.no_grad()
    def generate_with_trace(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "prompt": str,
            "response_text": str,
            "input_ids": List[int],
            "gen_ids": List[int],                # only new tokens
            "tokens": List[str],                 # decoded new tokens
            "chosen_logprobs": List[float],      # log p(y_t | context)
            "topk_logprobs": List[List[Tuple[int,float]]],  # per step [(tok_id, logp), ...]
            "scores_shape": (T, V) or None,      # debug
            "logits": optional raw logits (if enabled; big!)
          }
        """
        max_new_tokens = int(max_new_tokens or self.max_new_tokens)
        temperature = float(temperature if temperature is not None else self.temperature)
        top_p = float(top_p if top_p is not None else self.top_p)
        top_k = int(top_k if top_k is not None else self.top_k)
        do_sample = bool(self.do_sample if do_sample is None else do_sample)

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        # key bit: return_dict_in_generate + output_scores gives us per-step logits
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            return_dict_in_generate=True,
            output_scores=True,
        )
        seq_ids = out.sequences[0]             # [input_len + T]
        gen_ids = seq_ids[input_ids.shape[1]:] # [T] new tokens
        scores = out.scores                    # list of length T, each [1, V] logits

        # per-step chosen logprob + top-k logprobs
        chosen_logprobs: List[float] = []
        topk_logprobs: List[List[Tuple[int, float]]] = []
        raw_logits = [] if self.return_logits else None

        for t, logits in enumerate(scores):
            # logits: [1, V]
            if self.return_logits:
                raw_logits.append(logits.detach().cpu().float())

            # temperature was already applied in sampling; for numeric stability we compute log_softmax on given logits
            logprobs = torch.log_softmax(logits[0], dim=-1)  # [V]
            tok_id = int(gen_ids[t].item())
            chosen_logprobs.append(float(logprobs[tok_id].item()))

            # collect top-k table for SCM features
            if top_k > 0:
                vals, idx = torch.topk(logprobs, k=min(top_k, logprobs.shape[-1]))
                step_top = [(int(idx[i].item()), float(vals[i].item())) for i in range(vals.shape[0])]
                topk_logprobs.append(step_top)
            else:
                topk_logprobs.append([])

        # decode step tokens
        tokens = self.tokenizer.convert_ids_to_tokens(gen_ids.tolist(), skip_special_tokens=False)
        response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        trace: Dict[str, Any] = {
            "prompt": prompt,
            "response_text": response_text,
            "input_ids": input_ids[0].detach().cpu().tolist(),
            "gen_ids": gen_ids.detach().cpu().tolist(),
            "tokens": tokens,
            "chosen_logprobs": chosen_logprobs,
            "topk_logprobs": topk_logprobs,
            "scores_shape": (len(scores), int(scores[0].shape[-1])) if scores else (0, 0),
        }
        if raw_logits is not None:
            # warning: huge; keep off unless really required
            trace["logits"] = [x.numpy().tolist() for x in raw_logits]
        return trace
