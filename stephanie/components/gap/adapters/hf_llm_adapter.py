# stephanie/components/gap/adapters/hf_llm_adapter.py
from __future__ import annotations
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_TORCH_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

class HFLLMAdapter:
    """
    Minimal HuggingFace LLM adapter for the GAP component.

    API:
      - generate(prompt: str) -> str
      - score_heads(prompt: str, output: str) -> {dim: {"score": float, "rationale": str}}

    Notes:
      * MRQ scorer is resolved from `container` (or cfg) and expected to output 0..1 scores already.
      * Model/tokenizer are lazy-loaded on first use to keep startup fast.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        # --- config ---
        self.repo: str = self.cfg.get("repo", "")  # e.g. "mistralai/Mistral-7B-Instruct-v0.3"
        self.trust_remote_code: bool = bool(self.cfg.get("trust_remote_code", False))
        self.max_new_tokens: int = int(self.cfg.get("max_new_tokens", 128))
        self.temperature: float = float(self.cfg.get("temperature", 0.0))
        self.top_p: float = float(self.cfg.get("top_p", 1.0))
        self.do_sample: bool = bool(self.cfg.get("do_sample", False))
        self.dims: List[str] = list(self.cfg.get(
            "dims",
            ["reasoning","knowledge","clarity","faithfulness","coverage"]
        ))

        # dtype & device
        dtype_key = str(self.cfg.get("dtype", "float16")).lower()
        self.torch_dtype = _TORCH_DTYPES.get(dtype_key, torch.float16)
        self.device = self.cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = self.cfg.get("device_map", ("auto" if self.device == "cuda" else None))

        # optional extra kwargs for from_pretrained (quantization, etc.)
        self.model_kwargs: Dict[str, Any] = dict(self.cfg.get("model_kwargs", {}))

        # MRQ scorer resolution: try explicit cfg key, else container
        self.mrq = self._resolve_mrq_scorer()

        # lazy HF objects
        self._tok = None
        self._lm = None

        if self.logger:
            self.logger.log("HFAdapterInit", {
                "repo": self.repo,
                "device": self.device,
                "device_map": self.device_map,
                "dtype": str(self.torch_dtype),
                "dims": self.dims,
                "have_mrq": bool(self.mrq is not None),
            })

    # ---------------- internal helpers ----------------

    def _resolve_mrq_scorer(self):
        # 1) cfg override (a dotted path or container key you support in your app)
        mrq_key = self.cfg.get("mrq_key")
        if mrq_key and hasattr(self.container, mrq_key):
            return getattr(self.container, mrq_key)

        # 2) common container conventions (adapt to your service layout)
        for attr in ("mrq_scorer", "scoring_mrq", "scorer_mrq", "scoring", "services"):
            svc = getattr(self.container, attr, None)
            if svc is None:
                continue
            # direct scorer
            if hasattr(svc, "score") and hasattr(svc, "models"):
                return svc
            # nested
            for sub in ("mrq", "mrq_scorer"):
                inner = getattr(svc, sub, None)
                if inner is not None and hasattr(inner, "score"):
                    return inner

        return None

    def _ensure_model(self):
        if self._tok is not None and self._lm is not None:
            return
