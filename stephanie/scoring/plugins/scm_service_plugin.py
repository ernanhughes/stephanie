# stephanie/scoring/plugins/scm_service_plugin.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .registry import register


@register("scm")
class SCMServicePlugin:
    """
    Computes SCM features from existing LL stats in tap_output.attributes.
    Falls back to host._ll_stats(goal, resp) only if attrs are missing.
    Mirrors scm.{dim}.score01 into {alias}.{dim}.* without touching base vector.
    """
    def __init__(self, container=None, logger=None, host=None,
                 model_alias: Optional[str]=None, topk: int = 0,
                 ppl_low: float = 5.0, ppl_high: float = 40.0):
        self.container = container
        self.logger = logger
        self.host = host
        self.alias = model_alias or getattr(host, "model_alias", getattr(host, "model_type", "model"))
        self.topk = int(topk or 0)
        self.ppl_low = float(ppl_low)
        self.ppl_high = float(ppl_high)
        try:
            self.scm_svc = container.get("scm_service")
        except Exception:
            self.scm_svc = None

    def post_process(self, *, tap_output: Dict[str, Any]) -> Dict[str, float]:
        attrs = (tap_output.get("attributes") or {})
        goal  = tap_output.get("goal_text", "") or ""
        resp  = tap_output.get("resp_text", "") or ""

        # 1) Pull LL stats from attributes first (no recompute).
        stats = {}
        for k in ("mean_logprob", "ppl", "entropy_mean", "len_tokens",
                  "len_chars", "bytes_len", "sum_nll_nats", "bpb"):
            if k in attrs and self._finite(attrs[k]):
                stats[k] = float(attrs[k])

        # 2) If absent, use host._ll_stats(goal, resp) as fallback (HF scorer path).
        if (not stats) and self.host is not None and hasattr(self.host, "_ll_stats"):
            try:
                stats = self.host._ll_stats(goal, resp) or {}
            except Exception as e:
                stats = {}
                if self.logger: self.logger.log("SCMPlugin_ll_stats_error", {"error": str(e)})

        if not stats or not self.scm_svc:
            # Nothing to add.
            return {}

        # 3) Derive SCM from stats via service (single source of truth).
        try:
            vocab_size = None
            try:
                # Best-effort vocab probe for HF scorers
                if getattr(self.host, "tok", None) is not None:
                    vocab_size = getattr(self.host.tok, "vocab_size", None)
                    if vocab_size is None and hasattr(self.host.tok, "get_vocab"):
                        vocab_size = len(self.host.tok.get_vocab() or {})
            except Exception:
                pass

            scm = self.scm_svc.derive_scm_from_ll(
                stats,
                ppl_low=self.ppl_low,
                ppl_high=self.ppl_high,
                vocab_size=vocab_size
            )
        except Exception as e:
            if self.logger: self.logger.log("SCMServiceError", {"error": str(e)})
            return {}

        # (Optional) top-k token payload hook; keep out of vector
        if self.topk > 0 and hasattr(self.host, "token_topk"):
            try:
                _ = self.host.token_topk(goal, resp, k=self.topk)
            except Exception:
                pass

        # 4) Mirror per-dim scores under alias.* so downstream selectors find them.
        out: Dict[str, float] = {}
        out.update({k: float(v) for k, v in scm.items() if self._finite(v)})

        for dim in ("reasoning", "knowledge", "clarity", "faithfulness", "coverage"):
            k = f"scm.{dim}.score01"
            if k in scm and self._finite(scm[k]):
                v = float(scm[k])
                out[f"{self.alias}.{dim}.score01"]  = v
                out[f"{self.alias}.{dim}.score100"] = round(v * 100.0, 4)
                out[f"{self.alias}.{dim}"]          = v

        return out

    @staticmethod
    def _finite(x) -> bool:
        try: return math.isfinite(float(x))
        except Exception: return False
