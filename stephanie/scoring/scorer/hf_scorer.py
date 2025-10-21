# stephanie/scoring/scorer/huggingface_scorer.py
from __future__ import annotations

import gc
import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorer.base_scorer import BaseScorer


class HuggingFaceScorer(BaseScorer):
    """
    Simple, stable HF CausalLM scorer (Windows-friendly).
    - Computes teacher-forced LL/entropy stats of response conditioned on goal
    - Does *not* compute SCM or calibration; plugins add that after _score_core()
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger, enable_plugins=True)
        self.model_type = "hf"

        # --- config
        self.model_name = str(cfg.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
        self.tokenizer_name = cfg.get("tokenizer_name") or self.model_name
        self.max_seq_len = int(cfg.get("max_seq_len", 4096))
        self.device_map = cfg.get("device_map", "auto")
        self.trust_remote_code = bool(cfg.get("trust_remote_code", True))
        self.model_alias = str(cfg.get("model_alias", "hf"))
        self.dimensions: List[str] = cfg.get(
            "dimensions",
            ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"],
        )

        # Optional HF cache knobs
        self.cache_dir = cfg.get("cache_dir") or os.environ.get("HF_HOME") or None
        self.local_files_only = bool(cfg.get("local_files_only", False))

        # dtype selection
        dtype_str = str(cfg.get("torch_dtype", "auto"))
        if dtype_str == "auto":
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            self.torch_dtype = getattr(torch, dtype_str, torch.float32)

        # --- tokenizer (fast → slow fallback)
        tok_id = self.tokenizer_name
        try:
            self.tok = AutoTokenizer.from_pretrained(
                tok_id,
                use_fast=True,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        except Exception as e_fast:
            self.logger and self.logger.log("HFTokenizerFastFailed", {"model": tok_id, "error": str(e_fast)})
            self.tok = AutoTokenizer.from_pretrained(
                tok_id,
                use_fast=False,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )

        # Ensure pad token
        if not getattr(self.tok, "pad_token", None):
            try:
                self.tok.pad_token = self.tok.eos_token
            except Exception:
                pass

        # --- model (eager, stable)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self.model.eval()

        # Force eager attention (avoid SDPA/Flash issues on some Windows/CUDA combos)
        try:
            if hasattr(self.model, "config"):
                if getattr(self.model.config, "attn_implementation", None) is not None:
                    self.model.config.attn_implementation = "eager"
                setattr(self.model.config, "_attn_implementation", "eager")
        except Exception:
            pass

        # For uniformity with other scorers
        self.embedding_type = self.memory.embedding.name

        # safe device probe
        try:
            p = next(self.model.parameters())
            dev_str = str(p.device)
        except Exception:
            dev_str = "unknown"

        self.logger and self.logger.log(
            "HFScorerLoaded",
            {"model": self.model_name, "alias": self.model_alias, "device": dev_str, "dtype": str(self.torch_dtype)},
        )

    # -----------------------------
    # Core scoring (plugins will enhance results afterward)
    # -----------------------------
    def _score_core(self, context: dict, scorable, dimensions: List[str]) -> ScoreBundle:
        """
        Returns basic stats only. SCM, calibration, top-k, etc. are added by plugins.
        """
        goal_text = (context.get(GOAL, {}) or {}).get(GOAL_TEXT, "") or ""
        resp_text = scorable.text or ""

        with torch.no_grad():
            stats = self._ll_stats(goal_text, resp_text)

        # Minimal attributes that plugins can build on
        base_attrs = {
            "mean_logprob": stats["mean_logprob"],
            "ppl": stats["ppl"],
            "entropy_mean": stats["entropy_mean"],
            "len_tokens": stats["len_tokens"],
            "len_chars": stats["len_chars"],
            "bytes_len": stats["bytes_len"],
            "sum_nll_nats": stats["sum_nll_nats"],
            "bpb": stats["bpb"],
        }

        # Build a small vector under the model alias (no SCM keys here)
        vector = self._build_base_vector(self.model_alias, base_attrs)

        results: Dict[str, ScoreResult] = {}
        for dim in dimensions:
            # We don’t set a semantic score here; plugins may write scm.* to attributes.
            # Keep score=0.0 as placeholder so downstream code has a float.
            results[dim] = ScoreResult(
                dimension=dim,
                score=0.0,
                source=self.model_type,
                rationale=(
                    f"{self.model_alias}[{dim}] ppl={stats['ppl']:.2f}, "
                    f"H̄={stats['entropy_mean']:.3f}, lp̄={stats['mean_logprob']:.3f}"
                ),
                weight=1.0,
                attributes={**base_attrs, **vector},
            )

        return ScoreBundle(results=results)

    # -----------------------------
    # Optional helper for plugins (token distributions)
    # -----------------------------
    @torch.no_grad()
    def token_topk(self, goal: str, resp: str, k: int = 5) -> Optional[List[List[tuple[str, float]]]]:
        if not k or k <= 0:
            return None
        enc_goal = self.tok(goal, return_tensors="pt", add_special_tokens=False)
        enc_resp = self.tok(resp, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([enc_goal["input_ids"], enc_resp["input_ids"]], dim=1).to(self.model.device)
        attn = torch.ones_like(input_ids)
        out = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        shift_logits = out.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        start = enc_goal["input_ids"].shape[1]
        resp_logits = shift_logits[:, start:, :]  # [1, Lr, V]
        probs = torch.softmax(resp_logits, dim=-1)[0]  # [Lr, V]
        topv, topi = probs.topk(k, dim=-1)  # [Lr, k]
        toks = [self.tok.convert_ids_to_tokens(ids.tolist()) for ids in topi]
        return [[(toks[t][j], float(topv[t, j].item())) for j in range(k)]
                for t in range(topi.size(0))]

    # -----------------------------
    # Internals (teacher-forced stats only)
    # -----------------------------
    @torch.no_grad()
    def _ll_stats(self, goal: str, resp: str) -> Dict[str, float]:
        """
        Teacher-forced stats on response given goal.
        Returns: mean_logprob, ppl, entropy_mean, len_tokens, len_chars, bytes_len, bpb, sum_nll_nats
        """
        enc_goal = self.tok(goal, return_tensors="pt", add_special_tokens=False)
        enc_resp = self.tok(resp, return_tensors="pt", add_special_tokens=False)

        g_ids = enc_goal["input_ids"]
        r_ids = enc_resp["input_ids"]
        goal_len = g_ids.shape[1]
        total_len = goal_len + r_ids.shape[1]

        input_ids = torch.cat([g_ids, r_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        # left-truncate if too long, and adjust response start accordingly
        if total_len > self.max_seq_len:
            cut = total_len - self.max_seq_len
            input_ids = input_ids[:, cut:]
            attention_mask = attention_mask[:, cut:]
            resp_start = max(0, goal_len - cut)
        else:
            resp_start = goal_len

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # [B, T, V]

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # slice response portion after shift
        resp_logits = shift_logits[:, resp_start:, :]   # [B, Lr, V]
        resp_labels = shift_labels[:, resp_start:]      # [B, Lr]

        if resp_labels.numel() == 0:
            vocab_est = (
                getattr(self.tok, "vocab_size", None)
                or (len(getattr(self.tok, "get_vocab")() or {}) if hasattr(self.tok, "get_vocab") else None)
                or 32000
            )
            return dict(
                mean_logprob=0.0,
                ppl=float("inf"),
                entropy_mean=math.log(vocab_est),
                len_tokens=0,
                len_chars=len(resp),
                bytes_len=self._bytes_len(resp),
                sum_nll_nats=0.0,
                bpb=0.0,
            )

        logprobs = F.log_softmax(resp_logits, dim=-1)              # [B, Lr, V]
        chosen_lp = torch.gather(logprobs, dim=-1, index=resp_labels.unsqueeze(-1)).squeeze(-1)  # [B, Lr]
        mean_logprob = float(chosen_lp.mean().item())

        probs = logprobs.exp()
        ent = -(probs * logprobs).sum(dim=-1)                      # [B, Lr]
        entropy_mean = float(ent.mean().item())

        ppl = float(math.exp(-mean_logprob))

        sum_nll_nats = float(-chosen_lp.sum().item())  # total negative log-likelihood (nats)
        bytes_len = self._bytes_len(resp)
        bpb = self._to_bits(sum_nll_nats) / max(bytes_len, 1)      # bits per byte

        return dict(
            mean_logprob=mean_logprob,
            ppl=ppl,
            entropy_mean=entropy_mean,
            len_tokens=int(resp_labels.numel()),
            len_chars=len(resp),
            bytes_len=int(bytes_len),
            sum_nll_nats=sum_nll_nats,
            bpb=float(bpb),
        )

    # -----------------------------
    # Vector builder (base metrics only)
    # -----------------------------
    def _build_base_vector(self, alias: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            f"{alias}.mean_logprob",
            f"{alias}.ppl",
            f"{alias}.entropy_mean",
            f"{alias}.len_tokens",
            f"{alias}.len_chars",
            f"{alias}.bytes_len",
            f"{alias}.sum_nll_nats",
            f"{alias}.bpb",
        ]
        vec: Dict[str, float] = {}
        vec[f"{alias}.mean_logprob"] = float(attrs.get("mean_logprob", 0.0))
        vec[f"{alias}.ppl"] = float(attrs.get("ppl", float("inf")))
        vec[f"{alias}.entropy_mean"] = float(attrs.get("entropy_mean", 0.0))
        vec[f"{alias}.len_tokens"] = float(attrs.get("len_tokens", 0))
        vec[f"{alias}.len_chars"] = float(attrs.get("len_chars", 0))
        vec[f"{alias}.bytes_len"] = float(attrs.get("bytes_len", 0))
        vec[f"{alias}.sum_nll_nats"] = float(attrs.get("sum_nll_nats", 0.0))
        vec[f"{alias}.bpb"] = float(attrs.get("bpb", 0.0))

        cols = list(vec.keys())
        vals = [vec[c] for c in cols]
        return {"vector": vec, "columns": cols, "values": vals}

    # ---- helpers
    def _bytes_len(self, s: str) -> int:
        try:
            return len(s.encode("utf-8"))
        except Exception:
            return len(s)

    def _to_bits(self, nats: float) -> float:
        # bits = nats / ln(2)
        return float(nats / math.log(2.0))

    # Cleanup (also lets BaseScorer.close() clean up plugins)
    def close(self):
        try:
            # Detach hooks / peft / LoRA if any
            for attr in ("peft_model", "lora_model"):
                m = getattr(self, attr, None)
                if m is not None:
                    try:
                        m.cpu()
                    except Exception:
                        pass
                    setattr(self, attr, None)

            # Move to CPU first
            if getattr(self, "model", None) is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
            self.model = None

            # Tokenizer free
            if getattr(self, "tokenizer", None) is not None:
                self.tokenizer = None

            # Accelerate offload teardown
            try:
                offload = getattr(self, "_cpu_offload", None)
                if offload and hasattr(offload, "teardown"):
                    offload.teardown()
            except Exception:
                pass

            # GC + CUDA cache clear
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        finally:
            # Ensure plugins also get a chance to close
            super().close()
