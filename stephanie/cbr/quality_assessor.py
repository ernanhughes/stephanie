# stephanie/cbr/quality_assessor.py
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List


def _finite(x: float, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return x if (not math.isnan(x) and math.isfinite(x)) else default
        x = float(x)
        return x if math.isfinite(x) else default
    except Exception:
        return default

class DefaultQualityAssessor:
    """
    Aggregate a single quality scalar from:
      - MARS agreement
      - Preferred signal (configurable; defaults to HRM/alignment)
      - Optional LLM grade & reward proxy

    Config (all optional):
      quality_weights:
        mars: 1.0
        preferred: 0.5   # if missing, falls back to 'hrm' weight, then 0.5
        reward: 2.0
        llm: 0.25
      alignment_dimension: "alignment"   # used as a sensible default
      quality_preferred:
        source: "hrm"                    # str or [str,...] e.g. "contrastive_ranker"
        dimension: "hrm"                 # or "alignment" or ["alignment", "hrm"]
        attribute_field: null            # e.g. "final_score" or "llm_grade"
        source_match: "exact"            # "exact" | "substring"
    """

    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        self.qw = cfg.get("quality_weights", {}) or {
            "mars": 1.0, "hrm": 0.5, "reward": 2.0, "llm": 0.25
        }
        # weight for preferred: prefer explicit 'preferred', else reuse 'hrm', else 0.5
        self._w_preferred = _finite(self.qw.get("preferred", self.qw.get("hrm", 0.5)))
        self.alignment_dim = (cfg.get("alignment_dimension") or "alignment")

        # preferred signal selection (defaults target HRM-like entries)
        qp = cfg.get("quality_preferred", {}) or {}
        self.pref_source = qp.get("source", "hrm")
        self.pref_dimension = qp.get("dimension", "hrm")  # often "hrm" or "alignment"
        self.pref_attr = qp.get("attribute_field", None)
        self.pref_src_match = (qp.get("source_match") or "exact").lower()

    # ---- public API ----------------------------------------------------
    def quality(self, mars_results: Dict, scores_payload: Dict) -> float:
        mars_agree = self._mars_agreement(mars_results)

        preferred, llm_grade, reward = 0.0, 0.0, 0.0
        if scores_payload:
            preferred  = self._extract_preferred_mean(scores_payload)
            llm_grade  = self._extract_llm_grade_mean(scores_payload)
            reward     = self._extract_reward_proxy_mean(scores_payload)

        total = (
            _finite(self.qw.get("mars", 1.0)) * _finite(mars_agree)
            + self._w_preferred * _finite(preferred)
            + _finite(self.qw.get("llm", 0.25)) * _finite(llm_grade)
            + _finite(self.qw.get("reward", 2.0)) * _finite(reward)
        )

        self.logger and self.logger.log("QualityAssessorAggregate", {
            "mars_agreement": round(_finite(mars_agree), 6),
            "preferred_mean": round(_finite(preferred), 6),
            "llm_grade_mean": round(_finite(llm_grade), 6),
            "reward_proxy_mean": round(_finite(reward), 6),
            "weights": {"mars": self.qw.get("mars", 1.0),
                        "preferred": self._w_preferred,
                        "llm": self.qw.get("llm", 0.25),
                        "reward": self.qw.get("reward", 2.0)},
            "preferred_cfg": {
                "source": self.pref_source,
                "dimension": self.pref_dimension,
                "attribute_field": self.pref_attr,
                "source_match": self.pref_src_match,
            },
            "quality": round(_finite(total), 6),
        })
        return float(total)

    # ---- components ----------------------------------------------------
    def _mars_agreement(self, mars_results: Dict) -> float:
        if not mars_results:
            return 0.0
        vals: List[float] = [ _finite(v.get("agreement_score", 0.0)) for v in mars_results.values() ]
        return sum(vals)/len(vals) if vals else 0.0

    def _extract_preferred_mean(self, payload: Dict[str, Any]) -> float:
        """
        Select values matching your configured preferred signal. Default matches HRM:
        - by source == 'hrm' OR
        - by dimension == 'hrm' (or 'alignment').
        If 'attribute_field' is set, use that attribute; else use the per-result 'score'.
        """
        def _as_list(x) -> List[str]:
            if x is None:
                return []
            if isinstance(x, (list, tuple, set)):
                return [str(v).lower() for v in x]
            return [str(x).lower()]

        want_srcs = _as_list(self.pref_source) or ["hrm"]
        want_dims = _as_list(self.pref_dimension) or ["hrm", self.alignment_dim]

        vals: List[float] = []

        for _, bundle in (payload or {}).items():
            res_map = (bundle or {}).get("results") or {}
            for dim, r in res_map.items():
                src = str(r.get("source", "")).lower()
                dim_l = str(dim).lower()
                attrs = (r.get("attributes") or {})

                # match by source
                src_match = any(
                    (src == s) if self.pref_src_match == "exact" else (s in src)
                    for s in want_srcs
                ) if want_srcs else False

                # match by dimension
                dim_match = (dim_l in want_dims) if want_dims else False

                if src_match or dim_match:
                    val = attrs.get(self.pref_attr) if self.pref_attr else r.get("score")
                    vals.append(_finite(val))

        # Fallback: try common HRM-alikes if configured selection yielded nothing
        if not vals:
            for _, bundle in (payload or {}).items():
                res_map = (bundle or {}).get("results") or {}
                for dim, r in res_map.items():
                    src = str(r.get("source", "")).lower()
                    if src == "hrm" or dim in ("hrm", self.alignment_dim):
                        vals.append(_finite(r.get("score")))

        return sum(vals)/len(vals) if vals else 0.0

    # unchanged helpers below
    def _extract_llm_grade_mean(self, payload: Dict[str, Any]) -> float:
        vals = []
        for _, bundle in payload.items():
            res_map = (bundle or {}).get("results") or {}
            for _, r in res_map.items():
                src = str(r.get("source", "")).lower()
                attrs = (r.get("attributes") or {})
                if "llm" in src or "grader" in src:
                    vals.append(_finite(r.get("score")))
                elif "llm_grade" in attrs:
                    vals.append(_finite(attrs.get("llm_grade")))
                elif "grade" in attrs and src.startswith("llm"):
                    vals.append(_finite(attrs.get("grade")))
        return sum(vals)/len(vals) if vals else 0.0

    def _extract_reward_proxy_mean(self, payload: Dict[str, Any]) -> float:
        rewards = []
        sicql_qs = []
        for _, bundle in payload.items():
            res_map = (bundle or {}).get("results") or {}
            for _, r in res_map.items():
                src = str(r.get("source", "")).lower()
                attrs = (r.get("attributes") or {})
                if src in ("contrastive_ranker", "reward", "contrastive"):
                    rewards.append(_finite(r.get("score")))
                if src == "sicql" or "q_value" in attrs:
                    q = attrs.get("q_value", None)
                    if q is not None:
                        sicql_qs.append(_finite(q))
        if rewards:
            return sum(rewards)/len(rewards)
        if sicql_qs:
            return sum(self._squash01(q) for q in sicql_qs) / len(sicql_qs)
        return 0.0

    @staticmethod
    def _squash01(x: float) -> float:
        x = _finite(x, 0.0)
        y = x / (1.0 + abs(x))  # [-1,1]
        return 0.5 * (y + 1.0)  # [0,1]
