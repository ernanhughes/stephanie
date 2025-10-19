from __future__ import annotations

import math
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
    Generic HF CausalLM scorer that computes SCM-style metrics from teacher-forced
    log-likelihood/entropy of the response conditioned on the goal.

    cfg keys:
      model_name: str                        # e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
      tokenizer_name: str|None               # override if model card recommends different
      ppl_range: [low, high]                 # map perplexity -> ood_hat01
      max_seq_len: int                       # truncate safety
      device_map: "auto" | None
      torch_dtype: "auto" | "float16" | "bfloat16" | "float32"
      trust_remote_code: bool
      model_alias: "hf" | "hrm" | "tiny"     # vector prefix (default "hf")
      dimensions: [reasoning, knowledge, ...]
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
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
            ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]
        )
        self.ppl_low, self.ppl_high = (cfg.get("ppl_range") or [5.0, 40.0])[:2]

        # dtype
        dtype_str = str(cfg.get("torch_dtype", "auto"))
        if dtype_str == "auto":
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            self.torch_dtype = getattr(torch, dtype_str)

        # --- tokenizer (fast → slow)
        tok_id = self.tokenizer_name
        try:
            self.tok = AutoTokenizer.from_pretrained(
                tok_id, use_fast=True, trust_remote_code=self.trust_remote_code
            )
        except Exception as e_fast:
            self.logger and self.logger.log("HFTokenizerFastFailed", {"model": tok_id, "error": str(e_fast)})
            try:
                self.tok = AutoTokenizer.from_pretrained(
                    tok_id, use_fast=False, trust_remote_code=self.trust_remote_code
                )
            except Exception as e_slow:
                msg = (
                    f"Failed to load tokenizer for '{tok_id}'. "
                    "Install: pip install -U tokenizers sentencepiece protobuf. "
                    f"Fast error: {e_fast} | Slow error: {e_slow}"
                )
                raise RuntimeError(msg)

        # sensible pad token for causal LMs
        if not getattr(self.tok, "pad_token", None):
            try:
                self.tok.pad_token = self.tok.eos_token
            except Exception:
                pass

        # --- model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()

        # For uniformity with other scorers
        self.embedding_type = self.memory.embedding.name

        # safe device probe (works with device_map="auto")
        try:
            p = next(self.model.parameters())
            dev_str = str(p.device)
        except Exception:
            dev_str = "unknown"

        self._load_calibration()

        self.logger.log("HFScorerLoaded", {
            "model": self.model_name,
            "alias": self.model_alias,
            "device": dev_str,
            "dtype": str(self.torch_dtype),
        })


    # -----------------------------
    # Public scoring API
    # -----------------------------
    def score(self, context: dict, scorable, dimensions: List[str]) -> ScoreBundle:
        goal_text = (context.get(GOAL, {}) or {}).get(GOAL_TEXT, "") or ""
        resp_text = scorable.text or ""

        with torch.no_grad():
            stats = self._ll_stats(goal_text, resp_text)

        scm = self._scm_from_ll(stats)
        if self.cfg.get("compute_zscores", True) and getattr(self, "calib", None):
            try:
                mlp = stats["mean_logprob"]; ent = stats["entropy_mean"]; bpb = stats.get("bpb", float("nan"))
                c = self.calib  # expects means/stds for fields
                stats["z_mean_logprob"] = self._zs(mlp, c["mean_logprob"]["mean"], c["mean_logprob"]["std"])
                stats["z_entropy"]      = self._zs(ent, c["entropy_mean"]["mean"], c["entropy_mean"]["std"])
                if math.isfinite(bpb) and "bpb" in c:
                    stats["z_bpb"]     = self._zs(bpb, c["bpb"]["mean"], c["bpb"]["std"])
            except Exception as e:
                self.logger and self.logger.log("HFCalibApplyError", {"error": str(e)})

        k = int(self.cfg.get("expose_token_dists_topk", 0) or 0)
        if k > 0:
            try:
                stats["topk_per_token"] = self.token_topk(goal_text, resp_text, k=k)  # list[list[(tok, p)]]
            except Exception as e:
                self.logger and self.logger.log("HFTopKError", {"k": k, "error": str(e)})


        results: Dict[str, ScoreResult] = {}
        for dim in dimensions:
            v01 = float(scm.get(f"scm.{dim}.score01", 0.0))
            attrs = {
                **stats,
                **scm,
                f"{self.model_alias}.{dim}.score01": v01,
                f"{self.model_alias}.{dim}.score100": round(v01 * 100.0, 4),
                f"{self.model_alias}.{dim}": v01,
            }
            vector = self._build_vector(attrs)

            results[dim] = ScoreResult(
                dimension=dim,
                score=v01,
                source=self.model_type,
                rationale=(
                    f"{self.model_alias}[{dim}] ppl={stats['ppl']:.2f}, "
                    f"H̄={stats['entropy_mean']:.3f}, lp̄={stats['mean_logprob']:.3f}"
                ),
                weight=1.0,
                attributes={**attrs, **vector},
            )

        return ScoreBundle(results=results)

    @torch.no_grad()
    def token_topk(self, goal: str, resp: str, k: int = 5) -> Optional[List[List[tuple[str, float]]]]:
        if not k or k <= 0: return None
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
        # top-k per position
        topv, topi = probs.topk(k, dim=-1)  # [Lr, k]
        # decode tokens
        toks = [self.tok.convert_ids_to_tokens(ids.tolist()) for ids in topi]
        return [[(toks[t][j], float(topv[t, j].item())) for j in range(k)]
                for t in range(topi.size(0))]


    @staticmethod
    @torch.no_grad()
    def gap_metrics(a: HuggingFaceScorer, b: HuggingFaceScorer, goal: str, resp: str) -> Dict[str, float]:
        """
        Compute mean token-level JSD and directional KLs between two HF scorers
        on the response portion conditioned on goal.
        """
        eps = float(getattr(a.cfg, "eps", 1e-8))
        # encode once per scorer (tokenizers may differ; we align by each model's own tokens)
        def _resp_probs(scorer):
            tok = scorer.tok
            enc_g = tok(goal, return_tensors="pt", add_special_tokens=False)
            enc_r = tok(resp, return_tensors="pt", add_special_tokens=False)
            ids = torch.cat([enc_g["input_ids"], enc_r["input_ids"]], dim=1).to(scorer.model.device)
            attn = torch.ones_like(ids)
            out = scorer.model(input_ids=ids, attention_mask=attn, use_cache=False)
            sh_logits = out.logits[:, :-1, :]
            start = enc_g["input_ids"].shape[1]
            resp_logits = sh_logits[:, start:, :]          # [1, Lr, V]
            return torch.softmax(resp_logits, dim=-1)[0]   # [Lr, V]

        # NOTE: vocabularies differ → compare in each model's own space separately
        Pa = _resp_probs(a)  # [La, Va]
        Pb = _resp_probs(b)  # [Lb, Vb]

        # We summarize with each model's own mean entropy (already in stats), but for a symmetric gap,
        # compute JSD **within each model’s tokenization** between its probs and a smoothed projection
        # of the other model via temperatureless uniform prior. Practical, simple, defensible:
        # JSD(P || U) and JSD(Q || U) spike when either model is very “peaky” differently.
        # To keep it symmetric and bounded, do:
        Ua = torch.full_like(Pa, 1.0 / Pa.size(-1))
        Ub = torch.full_like(Pb, 1.0 / Pb.size(-1))

        def _kl(p, q):
            p = torch.clamp(p, eps, 1.0); q = torch.clamp(q, eps, 1.0)
            return (p * (p / q).log()).sum(dim=-1)

        def _jsd(p, q):
            m = 0.5 * (p + q)
            return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

        # two views (each in its own vocab space)
        jsd_a = _jsd(Pa, Ua).mean().item()
        jsd_b = _jsd(Pb, Ub).mean().item()

        # As an additional, model-agnostic scalar, compare **sequence log-likelihoods** (no vocab alignment needed)
        # Pull from each scorer's _ll_stats:
        Sa = a._ll_stats(goal, resp)
        Sb = b._ll_stats(goal, resp)
        d_mean_lp = float(Sa["mean_logprob"] - Sb["mean_logprob"])
        d_bpb     = float(Sa.get("bpb", float("nan")) - Sb.get("bpb", float("nan")))

        return {
            "gap_jsd_a": float(jsd_a),  # JSD in A's token space vs uniform
            "gap_jsd_b": float(jsd_b),  # JSD in B's token space vs uniform
            "gap_jsd_mean": float(0.5 * (jsd_a + jsd_b)),
            "delta_mean_logprob": d_mean_lp,
            "delta_bpb": d_bpb,
        }

    # -----------------------------
    # Internals
    # -----------------------------
    @torch.no_grad()
    def _ll_stats(self, goal: str, resp: str) -> Dict[str, float]:
        """
        Teacher-forced stats on response given goal.
        Returns:
          mean_logprob, ppl, entropy_mean, len_tokens, len_chars
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
            vocab_est = (getattr(self.tok, "vocab_size", None)
                         or (len(getattr(self.tok, "get_vocab")() or {}) if hasattr(self.tok, "get_vocab") else None)
                         or 32000)
            return dict(
                mean_logprob=0.0,
                ppl=float("inf"),
                entropy_mean=math.log(vocab_est),
                len_tokens=0,
                len_chars=len(resp),
            )

        logprobs = F.log_softmax(resp_logits, dim=-1)              # [B, Lr, V]
        chosen_lp = torch.gather(logprobs, dim=-1, index=resp_labels.unsqueeze(-1)).squeeze(-1)  # [B, Lr]
        mean_logprob = float(chosen_lp.mean().item())

        probs = logprobs.exp()
        ent = -(probs * logprobs).sum(dim=-1)                      # [B, Lr]
        entropy_mean = float(ent.mean().item())

        ppl = float(math.exp(-mean_logprob))

        # at end of _ll_stats()
        sum_nll_nats = float(-chosen_lp.sum().item())  # total negative log-likelihood (nats)
        bytes_len = self._bytes_len(resp)
        bpb = self._to_bits(sum_nll_nats) / max(bytes_len, 1)  # bits per byte

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

    def _scm_from_ll(self, st: Dict[str, float]) -> Dict[str, float]:
        ood = self._norm01(st["ppl"], self.ppl_low, self.ppl_high)
        ood_hat01 = float(max(0.0, min(1.0, ood)))

        vocab_est = (getattr(self.tok, "vocab_size", None)
                     or (len(getattr(self.tok, "get_vocab")() or {}) if hasattr(self.tok, "get_vocab") else None)
                     or 32000)
        ent_norm = st["entropy_mean"] / max(math.log(vocab_est), 1e-6)
        uncertainty01 = float(max(0.0, min(1.0, ent_norm)))

        lp01 = self._sig01(st["mean_logprob"], center=-1.5, scale=2.0)
        consistency01 = float(max(0.0, min(1.0, 0.6 * lp01 + 0.4 * (1.0 - uncertainty01))))
        length_norm01 = float(self._norm01(st["len_tokens"], 5.0, 200.0))
        temp01 = uncertainty01
        agree_hat01 = lp01

        reasoning = 0.55 * consistency01 + 0.35 * (1.0 - uncertainty01) + 0.10 * agree_hat01
        knowledge = 0.55 * (1.0 - ood_hat01) + 0.25 * lp01 + 0.20 * (1.0 - uncertainty01)
        clarity   = 0.50 * (1.0 - length_norm01) + 0.30 * (1.0 - uncertainty01) + 0.20 * consistency01
        faithful  = 0.45 * lp01 + 0.35 * consistency01 + 0.20 * (1.0 - uncertainty01)
        coverage  = 0.50 * (1.0 - ood_hat01) + 0.25 * (1.0 - uncertainty01) + 0.25 * length_norm01

        def clamp01(x: float) -> float: return float(min(1.0, max(0.0, x)))

        dim_scores = {
            "reasoning": clamp01(reasoning),
            "knowledge": clamp01(knowledge),
            "clarity": clamp01(clarity),
            "faithfulness": clamp01(faithful),
            "coverage": clamp01(coverage),
        }

        scm: Dict[str, float] = {f"scm.{k}.score01": v for k, v in dim_scores.items()}
        scm["scm.aggregate01"]   = float(sum(dim_scores.values()) / 5.0)
        scm["scm.uncertainty01"] = uncertainty01
        scm["scm.ood_hat01"]     = ood_hat01
        scm["scm.consistency01"] = consistency01
        scm["scm.length_norm01"] = length_norm01
        scm["scm.temp01"]        = temp01
        scm["scm.agree_hat01"]   = agree_hat01
        return scm

    def _build_vector(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            f"{self.model_alias}.mean_logprob",
            f"{self.model_alias}.ppl",
            f"{self.model_alias}.entropy_mean",
            f"{self.model_alias}.len_tokens",
            "scm.reasoning.score01","scm.knowledge.score01","scm.clarity.score01",
            "scm.faithfulness.score01","scm.coverage.score01","scm.aggregate01",
            "scm.uncertainty01","scm.ood_hat01","scm.consistency01",
            "scm.length_norm01","scm.temp01","scm.agree_hat01",
        ]
        vec: Dict[str, float] = {}
        vec[f"{self.model_alias}.mean_logprob"] = float(attrs.get("mean_logprob", 0.0))
        vec[f"{self.model_alias}.ppl"] = float(attrs.get("ppl", float("inf")))
        vec[f"{self.model_alias}.entropy_mean"] = float(attrs.get("entropy_mean", 0.0))
        vec[f"{self.model_alias}.len_tokens"] = float(attrs.get("len_tokens", 0))

        for k in keys[4:]:
            if k in attrs:
                vec[k] = float(attrs[k])

        for d in ["reasoning","knowledge","clarity","faithfulness","coverage"]:
            k = f"scm.{d}.score01"
            if k in attrs:
                v01 = float(attrs[k])
                vec[f"{self.model_alias}.{d}.score01"]  = v01
                vec[f"{self.model_alias}.{d}.score100"] = round(v01 * 100.0, 4)
                vec[f"{self.model_alias}.{d}"]          = v01

        cols = list(vec.keys())
        vals = [vec[c] for c in cols]
        return {"vector": vec, "columns": cols, "values": vals}


    def _bytes_len(self, s: str) -> int:
        try:    return len(s.encode("utf-8"))
        except: return len(s)

    def _to_bits(self, nats: float) -> float:
        return float(nats / math.log(2.0))

    def _load_calibration(self):
        self.calib = None
        path = self.cfg.get("calibration_path")
        if not path: return
        try:
            import json, os
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.calib = json.load(f) or None
                self.logger and self.logger.log("HFCalibLoaded", {"path": path})
        except Exception as e:
            self.logger and self.logger.log("HFCalibLoadError", {"error": str(e), "path": path})

    def _zs(self, x: float, mean: float, std: float) -> float:
        return float((x - mean) / (std if std > 1e-9 else 1.0))

    @staticmethod
    def _norm01(x: float, lo: float, hi: float) -> float:
        if not math.isfinite(x): return 1.0
        if hi <= lo: return 0.0
        return (x - lo) / (hi - lo)

    @staticmethod
    def _sig01(x: float, center: float = 0.0, scale: float = 1.0) -> float:
        z = (x - center) / max(scale, 1e-6)
        return 1.0 / (1.0 + math.exp(-z))

