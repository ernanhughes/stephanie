# stephanie/components/gap/services/scm_service.py
from __future__ import annotations

import math
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from stephanie.services.service_protocol import Service

__all__ = [
    "SCMService",
    "SCM_FEATURE_KEYS",
]

# Fixed order of Shared Core Metrics all models should emit
SCM_FEATURE_KEYS: List[str] = [
    "scm.reasoning.score01",
    "scm.knowledge.score01",
    "scm.clarity.score01",
    "scm.faithfulness.score01",
    "scm.coverage.score01",
    "scm.aggregate01",
    "scm.uncertainty01",
    "scm.ood_hat01",
    "scm.consistency01",
    "scm.length_norm01",
    "scm.temp01",
    "scm.agree_hat01",
]

_logger = logging.getLogger(__name__)


class SCMService(Service):
    """
    End-to-end SCM helper for HF-like scorers.

    Responsibilities:
      • `ll_stats(...)` – teacher-forced token stats (mean_logprob, ppl, entropy_mean, etc.)
      • `derive_scm_from_ll(stats, ppl_low, ppl_high, vocab_size)` – deterministic SCM features from stats
      • `from_ll_stats(stats, model_alias, ppl_range, vocab_size)` – one-call SCM dict (what plugins want)
      • `apply_calibration(stats, calib)` – optional z-scores
      • `build_vector(model_alias, attrs, dims)` – stable vector packing for downstream alignment
      • `extract_raw_per_dim(vector, model_label, dims)` – copy-through (no normalization)

    Notes:
      - All values returned here are *raw* (already in 0–1 where appropriate).
      - No extra normalization or range squeezing is applied beyond the formulas you own.
      - This service is stateless except for the device hint; safe to reuse across scorers.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._initialized: bool = False
        self.in_dim: int = len(SCM_FEATURE_KEYS)

    # ---- lifecycle ---------------------------------------------------------

    def initialize(self, **kwargs) -> None:
        """Initialize the service (kept for parity with other services)."""
        self._initialized = True
        _logger.info("SCMService initialized (in_dim=%d, device=%s)", self.in_dim, self.device)

    def shutdown(self) -> None:
        _logger.debug("SCMService shutdown complete")

    @property
    def name(self) -> str:
        return "scm-service-v1"

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "in_dim": self.in_dim,
            "device": str(self.device),
        }

    # ---- main steps --------------------------------------------------------


    # ---- plugin hook -------------------------------------------------------
    def post_process(self, *, tap_output: Dict[str, Any]) -> Dict[str, float]:
        """
        Generic scorer plugin entrypoint.
        Expected tap_output fields (BaseScorer supplies these):
          - goal_text: str
          - resp_text: str
          - host: scorer instance (HuggingFaceScorer, etc.)
          - model_alias: str (optional hint)
        Returns:
          Flat dict of { "scm.*": float } suitable for BaseScorer to merge.
        """
        # acquire host
        host = tap_output.get("host", None) or getattr(self, "host", None)
        if host is None:
            _logger.warning("SCMService.post_process: missing 'host' in tap_output; cannot compute ll_stats.")
            return {}

        goal = tap_output.get("goal_text", "") or ""
        resp = tap_output.get("resp_text", "") or ""
        if not (goal or resp):
            return {}  # nothing to score

        # Pull model bits from host
        tok = getattr(host, "tok", None)
        model = getattr(host, "model", None)
        max_seq_len = int(getattr(host, "max_seq_len", 4096))
        if tok is None or model is None:
            _logger.warning("SCMService.post_process: host lacks tok/model; aborting.")
            return {}

        # Compute teacher-forced stats
        try:
            stats = self.ll_stats(model, tok, goal, resp, max_seq_len)
        except Exception as e:
            _logger.warning("SCMService.post_process: ll_stats failed: %s", e)
            return {}

        # Optional calibration from host
        calib = getattr(host, "calib", None)
        try:
            self.apply_calibration(stats, calib)
        except Exception:
            pass

        # ppl_range: plugin opts > host.cfg > default
        opts = getattr(self, "_plugin_opts", {}) or {}
        pr = opts.get("ppl_range", None)
        if isinstance(pr, (list, tuple)) and len(pr) >= 2:
            ppl_low, ppl_high = float(pr[0]), float(pr[1])
        else:
            pr2 = (getattr(host, "cfg", {}) or {}).get("ppl_range", [5.0, 40.0])
            ppl_low, ppl_high = float(pr2[0]), float(pr2[1])

        vocab_size = self._safe_vocab(tok)

        scm = self.derive_scm_from_ll(
            stats, ppl_low=ppl_low, ppl_high=ppl_high, vocab_size=vocab_size
        )

        # Optional token-topk (don’t bloat output; you can add a metric for length if desired)
        k = int(opts.get("expose_token_dists_topk", 0) or 0)
        if k > 0 and hasattr(host, "token_topk"):
            try:
                _ = host.token_topk(goal, resp, k=k)
            except Exception:
                pass

        # return only scalar floats
        return {k: float(v) for k, v in scm.items()}

    @torch.no_grad()
    def ll_stats(
        self,
        model,
        tok,
        goal: str,
        resp: str,
        max_seq_len: int,
    ) -> Dict[str, float]:
        """
        Teacher-forced stats on response given goal.

        Returns keys:
          mean_logprob, ppl, entropy_mean, len_tokens, len_chars, bytes_len,
          sum_nll_nats, bpb
        """
        enc_goal = tok(goal, return_tensors="pt", add_special_tokens=False)
        enc_resp = tok(resp, return_tensors="pt", add_special_tokens=False)

        g_ids = enc_goal["input_ids"]
        r_ids = enc_resp["input_ids"]
        goal_len = g_ids.shape[1]
        total_len = goal_len + r_ids.shape[1]

        input_ids = torch.cat([g_ids, r_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        # Left-truncate if too long, adjust response start
        if total_len > max_seq_len:
            cut = total_len - max_seq_len
            input_ids = input_ids[:, cut:]
            attention_mask = attention_mask[:, cut:]
            resp_start = max(0, goal_len - cut)
        else:
            resp_start = goal_len

        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # [B, T, V]

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Slice response portion after shift (teacher forcing)
        resp_logits = shift_logits[:, resp_start:, :]   # [B, Lr, V]
        resp_labels = shift_labels[:, resp_start:]      # [B, Lr]

        if resp_labels.numel() == 0:
            vocab_est = self._safe_vocab(tok)
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

        logprobs = F.log_softmax(resp_logits, dim=-1)  # [B, Lr, V]
        chosen_lp = torch.gather(logprobs, dim=-1, index=resp_labels.unsqueeze(-1)).squeeze(-1)  # [B, Lr]
        mean_logprob = float(chosen_lp.mean().item())

        probs = logprobs.exp()
        ent = -(probs * logprobs).sum(dim=-1)          # [B, Lr]
        entropy_mean = float(ent.mean().item())

        ppl = float(math.exp(-mean_logprob))

        sum_nll_nats = float(-chosen_lp.sum().item())
        bytes_len = self._bytes_len(resp)
        bpb = self._to_bits(sum_nll_nats) / max(bytes_len, 1)

        return {
            "mean_logprob": mean_logprob,
            "ppl": ppl,
            "entropy_mean": entropy_mean,
            "len_tokens": int(resp_labels.numel()),
            "len_chars": len(resp),
            "bytes_len": int(bytes_len),
            "sum_nll_nats": sum_nll_nats,
            "bpb": float(bpb),
        }

    # Core SCM formula (unchanged from your HF scorer, centralized here)
    def derive_scm_from_ll(
        self,
        st: Dict[str, float],
        *,
        ppl_low: float,
        ppl_high: float,
        vocab_size: int | None,
    ) -> Dict[str, float]:
        ood = self._norm01(st["ppl"], ppl_low, ppl_high)
        ood_hat01 = float(self._clip01(ood))

        vocab_est = vocab_size or 32000
        ent_norm = st["entropy_mean"] / max(math.log(vocab_est), 1e-6)
        uncertainty01 = float(self._clip01(ent_norm))

        lp01 = self._sig01(st["mean_logprob"], center=-1.5, scale=2.0)
        consistency01 = float(self._clip01(0.6 * lp01 + 0.4 * (1.0 - uncertainty01)))
        length_norm01 = float(self._norm01(st["len_tokens"], 5.0, 200.0))
        temp01 = uncertainty01
        agree_hat01 = lp01

        def c01(x: float) -> float: return float(self._clip01(x))

        reasoning    = c01(0.55 * consistency01 + 0.35 * (1.0 - uncertainty01) + 0.10 * agree_hat01)
        knowledge    = c01(0.55 * (1.0 - ood_hat01) + 0.25 * lp01 + 0.20 * (1.0 - uncertainty01))
        clarity      = c01(0.50 * (1.0 - length_norm01) + 0.30 * (1.0 - uncertainty01) + 0.20 * consistency01)
        faithfulness = c01(0.45 * lp01 + 0.35 * consistency01 + 0.20 * (1.0 - uncertainty01))
        coverage     = c01(0.50 * (1.0 - ood_hat01) + 0.25 * (1.0 - uncertainty01) + 0.25 * length_norm01)

        dim_scores = {
            "reasoning": reasoning,
            "knowledge": knowledge,
            "clarity": clarity,
            "faithfulness": faithfulness,
            "coverage": coverage,
        }

        scm = {f"scm.{k}.score01": v for k, v in dim_scores.items()}
        scm["scm.aggregate01"]   = float(sum(dim_scores.values()) / 5.0)
        scm["scm.uncertainty01"] = uncertainty01
        scm["scm.ood_hat01"]     = ood_hat01
        scm["scm.consistency01"] = consistency01
        scm["scm.length_norm01"] = length_norm01
        scm["scm.temp01"]        = temp01
        scm["scm.agree_hat01"]   = agree_hat01
        return scm

    # One-call function the plugin can use directly
    def from_ll_stats(
        self,
        stats: Dict[str, float],
        *,
        model_alias: str,
        ppl_range: Tuple[float, float] = (5.0, 40.0),
        vocab_size: Optional[int] = None,
    ) -> Dict[str, float]:
        ppl_low, ppl_high = ppl_range
        return self.derive_scm_from_ll(stats, ppl_low=ppl_low, ppl_high=ppl_high, vocab_size=vocab_size)

    def apply_calibration(self, stats: Dict[str, float], calib: Optional[Dict[str, Dict[str, float]]]) -> None:
        """
        Mutates 'stats' to add z_* keys if calibration config is present.
        calib format: {name: {"mean": float, "std": float}, ...}
        """
        if not calib:
            return

        def _zs(x: float, mean: float, std: float) -> float:
            return float((x - mean) / (std if std > 1e-9 else 1.0))

        try:
            mlp = stats.get("mean_logprob", float("nan"))
            ent = stats.get("entropy_mean", float("nan"))
            bpb = stats.get("bpb", float("nan"))

            def _zmaybe(key: str, x: float) -> float:
                cfg = calib.get(key, None)
                if not cfg:
                    return float("nan")
                return _zs(x, cfg.get("mean", 0.0), cfg.get("std", 1.0))

            stats["z_mean_logprob"] = _zmaybe("mean_logprob", mlp)
            stats["z_entropy"]      = _zmaybe("entropy_mean", ent)
            if math.isfinite(bpb) and "bpb" in calib:
                stats["z_bpb"] = _zmaybe("bpb", bpb)
        except Exception:
            # keep silent; callers can log if desired
            pass

    def build_vector(self, *, model_alias: str, attrs: Dict[str, Any], dimensions: List[str]) -> Dict[str, Any]:
        """
        Assemble a stable vector with raw HF stats + scm.* + per-dim model_alias.* mirrors.
        """
        keys = [
            f"{model_alias}.mean_logprob",
            f"{model_alias}.ppl",
            f"{model_alias}.entropy_mean",
            f"{model_alias}.len_tokens",
            # SCM
            *SCM_FEATURE_KEYS,
        ]

        vec: Dict[str, float] = {
            f"{model_alias}.mean_logprob": float(attrs.get("mean_logprob", 0.0)),
            f"{model_alias}.ppl": float(attrs.get("ppl", float("inf"))),
            f"{model_alias}.entropy_mean": float(attrs.get("entropy_mean", 0.0)),
            f"{model_alias}.len_tokens": float(attrs.get("len_tokens", 0)),
        }

        for k in SCM_FEATURE_KEYS:
            if k in attrs:
                try:
                    vec[k] = float(attrs[k])
                except Exception:
                    pass

        # Mirror per-dimension 0–1 scores under model_alias.*
        for d in dimensions:
            k = f"scm.{d}.score01"
            if k in attrs:
                try:
                    v01 = float(attrs[k])
                    vec[f"{model_alias}.{d}.score01"]  = v01
                    vec[f"{model_alias}.{d}.score100"] = round(v01 * 100.0, 4)
                    vec[f"{model_alias}.{d}"]          = v01
                except Exception:
                    pass

        cols = list(vec.keys())
        vals = [vec[c] for c in cols]
        return {"vector": vec, "columns": cols, "values": vals}

    # ---- raw per-dim extractor (regex-free, suffix-safe) -------------------

    def extract_raw_per_dim(self, *, vector: Dict[str, float], model_label: str, dims: List[str]) -> Dict[str, float]:
        """
        For each dim in dims, copy:
          {prefix}.{dim}.score01 / score100 / score / aggregate*
          {prefix}.{dim}.attr.scm.* -> exposed as {prefix}.{dim}.scm.*
        """
        out: Dict[str, float] = {}
        prefix = self._detect_model_prefix(vector, model_label)

        def _take(k: str):
            if k in vector:
                try:
                    out[k] = float(vector[k])
                except Exception:
                    pass

        for dim in dims:
            for k in (
                f"{prefix}.{dim}.score01",
                f"{prefix}.{dim}.score100",
                f"{prefix}.{dim}.score",
                f"{prefix}.{dim}.aggregate01",
                f"{prefix}.{dim}.aggregate",
            ):
                _take(k)

            scm_root = f"{prefix}.{dim}.attr.scm."
            for k, v in vector.items():
                if isinstance(k, str) and k.startswith(scm_root):
                    tail = k[len(scm_root) :]
                    try:
                        out[f"{prefix}.{dim}.scm.{tail}"] = float(v)
                    except Exception:
                        pass

        return out

    # ---- utils -------------------------------------------------------------

    @staticmethod
    def _bytes_len(s: str) -> int:
        try:
            return len(s.encode("utf-8"))
        except Exception:
            return len(s)

    @staticmethod
    def _to_bits(nats: float) -> float:
        # bits = nats / ln(2)
        return float(nats / math.log(2.0))

    @staticmethod
    def _norm01(x: float, lo: float, hi: float) -> float:
        if not math.isfinite(x):
            return 1.0
        if hi <= lo:
            return 0.0
        return (x - lo) / (hi - lo)

    @staticmethod
    def _sig01(x: float, center: float = 0.0, scale: float = 1.0) -> float:
        z = (x - center) / max(scale, 1e-6)
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def _clip01(x: float) -> float:
        return min(1.0, max(0.0, float(x)))

    @staticmethod
    def _safe_vocab(tok) -> int:
        try:
            if getattr(tok, "vocab_size", None):
                return int(tok.vocab_size)
            if hasattr(tok, "get_vocab"):
                v = tok.get_vocab() or {}
                return int(len(v))
        except Exception:
            pass
        return 32000

    @staticmethod
    def _detect_model_prefix(vec: dict, fallback: str) -> str:
        roots = {k.split(".", 1)[0] for k in vec.keys() if isinstance(k, str) and "." in k}
        if fallback in roots:
            return fallback
        # Prefer the longest root that contains the fallback text (e.g., "hf_hrm" when fallback="hrm")
        for r in sorted(roots, key=len, reverse=True):
            if fallback in r:
                return r
        return (sorted(roots)[0] if roots else fallback)
