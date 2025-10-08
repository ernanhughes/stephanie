# stephanie/cbr/adaptor.py
"""
DefaultAdaptor (hybrid)
=======================
A robust adaptor that combines your DSPy-based reviser with a safe
heuristic fallback. It keeps your current public interface intact while adding:

- **DSPy path** (if available): uses your `AdaptSignature` and `Predict`.
- **Optional LLM callable**: if you provide `llm=callable`, it can be tried
  (before or after DSPy) with a prompt; validated against goal coverage &
  basic constraints.
- **Heuristic merge**: dependency-free improvements using keyword coverage
  from the donor (retrieved) text, with length controls and dedupe.
- **Rationales & mode**: returns what was done and why, including `mode`.

Public API (unchanged):
    adaptor = DefaultAdaptor(cfg, logger=logger, llm=my_llm_optional)
    res = adaptor.revise(goal_text=..., candidate_text=..., retrieved_text=...)
    # -> {"revised": str, "rationale": str, "mode": "dspy"|"llm"|"heuristic"|"bypass"}

Config keys (examples):
    {
      "enabled": true,
      "use_dspy_first": true,
      "use_llm_first": true,
      "validation": {"coverage_eps": 0.01, "max_len": 1200, "max_sents": 12},
      "merge_strategy": "append_best",  # or "replace_worst"
      "donor_sent_limit": 2,
      "top_goal_kws": 12,
      "min_kw_len": 4,
    }
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# -------- Optional DSPy import --------
_HAS_DSPY = True
try:
    from dspy import InputField, OutputField, Predict, Signature
except Exception:  # pragma: no cover
    _HAS_DSPY = False
    InputField = OutputField = Predict = Signature = object  # type: ignore


# -------- System prompt (used for LLM prompt; DSPy can be configured externally) --------
_ADAPT_SYSTEM_PROMPT = (
    "You are an expert editor. Improve the candidate so it better satisfies the goal. "
    "Use helpful, factual ideas from the donor text when relevant. "
    "Be concise, neutral, and self-contained. Return only the revised content."
)


# -------- DSPy signature (only defined when DSPy is available) --------
if _HAS_DSPY:
    class AdaptSignature(Signature):
        goal      = InputField(desc="Goal text (what we are trying to achieve)")
        candidate = InputField(desc="Base candidate text to improve")
        retrieved = InputField(desc="Relevant retrieved case text to adapt from")
        revised   = OutputField(desc="Revised/improved candidate that better fits the goal")
        rationale = OutputField(desc="Brief explanation of what was changed and why")
else:
    AdaptSignature = None  # type: ignore


# -------- Config dataclass --------
@dataclass
class DefaultAdaptorCfg:
    enabled: bool = True
    # ordering of strategies
    use_dspy_first: bool = True
    use_llm_first: bool = True
    # validation thresholds
    coverage_eps: float = 0.01  # require at least this much goal-coverage gain
    max_chars: int = 1200
    min_chars: int = 120
    max_sentences: int = 12
    min_sentences: int = 3
    # keyword & merge controls
    min_kw_len: int = 4
    top_goal_kws: int = 12
    donor_sent_limit: int = 2
    merge_strategy: str = "append_best"  # or "replace_worst"
    top_k: int = 12
    improv_eps: float = 0.1  # require at least this much improvement in goal coverage
    scorer_name: str = "sicql"

# -------- Adaptor implementation --------
class DefaultAdaptor:
    """Hybrid adaptor that tries DSPy, then optional LLM, then heuristic."""

    def __init__(self, cfg: Dict[str, Any] | None = None, logger=None, llm: Optional[Callable[[str], str]] = None):
        self.cfg = DefaultAdaptorCfg(**(cfg or {}))
        self.logger = logger or logging.getLogger("default-adaptor")
        self.llm = llm  # optional prompt-based callable
        self._predict = None
        if _HAS_DSPY and self.cfg.use_dspy_first and AdaptSignature is not None:
            try:
                self._predict = Predict(AdaptSignature)  # relies on dspy.configure(lm=...)
            except Exception as e:
                self._predict = None
                self._log("AdaptInitDSPyError", error=str(e))

    # ---- public API ----
    def revise(self, *, goal_text: str, candidate_text: str, retrieved_text: str) -> Dict[str, str]:
        goal = (goal_text or "").strip()
        cand = (candidate_text or "").strip()
        donor = (retrieved_text or "").strip()

        if not self.cfg.enabled:
            return {"revised": cand, "rationale": "Adaptor disabled", "mode": "bypass"}

        # Baseline measurements
        base_cov = self._coverage(goal, cand)

        # 1) Try DSPy path first
        if self._predict is not None:
            try:
                out = self._predict(goal=goal, candidate=cand, retrieved=donor)
                cand_dspy = (getattr(out, "revised", "") or "").strip()
                rationale = (getattr(out, "rationale", "") or "").strip()
                ok, why = self._accept(goal, cand, cand_dspy, base_cov)
                if ok:
                    cand_dspy = self._finalize_text(cand_dspy)
                    return {"revised": cand_dspy, "rationale": rationale or why, "mode": "dspy"}
                else:
                    self._log("AdaptDSPyRejected", reason=why)
            except Exception as e:
                self._log("AdaptDSPyError", error=str(e))

        # 2) Try external LLM callable
        if self.llm and self.cfg.use_llm_first:
            try:
                prompt = self._build_prompt(goal, cand, donor)
                out = (self.llm(prompt) or "").strip()
                ok, why = self._accept(goal, cand, out, base_cov)
                if ok:
                    out = self._finalize_text(out)
                    return {"revised": out, "rationale": "LLM rewrite accepted: " + why, "mode": "llm"}
                else:
                    self._log("AdaptLLMRejected", reason=why)
            except Exception as e:
                self._log("AdaptLLMError", error=str(e))

        # 3) Heuristic merge (deterministic, dependency-free)
        revised, reasons = self._heuristic_merge(goal, cand, donor)
        revised = self._finalize_text(revised)
        return {"revised": revised, "rationale": "; ".join(reasons) or "heuristic_merge", "mode": "heuristic"}

    # ---- acceptance & validation ----
    def _accept(self, goal: str, old: str, new: str, base_cov: float) -> Tuple[bool, str]:
        new = (new or "").strip()
        if not new:
            return False, "empty_output"
        # sentence/length constraints
        sents = self._split_sentences(new)
        if len(sents) > self.cfg.max_sentences:
            return False, "too_many_sentences"
        if len(sents) < self.cfg.min_sentences:
            return False, "too_few_sentences"
        if len(new) > self.cfg.max_chars:
            return False, "too_long"
        if len(new) < self.cfg.min_chars:
            return False, "too_short"
        # goal coverage improvement
        cov_new = self._coverage(goal, new)
        if cov_new < base_cov + self.cfg.coverage_eps:
            return False, f"no_coverage_gain({cov_new:.3f}<= {base_cov + self.cfg.coverage_eps:.3f})"
        return True, f"coverage_gain {cov_new - base_cov:.3f}"

    # ---- heuristic merge ----
    def _heuristic_merge(self, goal: str, cand: str, donor: str) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        if not donor:
            reasons.append("no_donor_available")
            return cand, reasons

        goal_kws = self._top_keywords(goal, k=self.cfg.top_goal_kws)
        cand_kws = set(self._top_keywords(cand, k=self.cfg.top_goal_kws))
        donor_sents = self._split_sentences(donor)
        cand_sents = self._split_sentences(cand)

        missing = set(goal_kws) - cand_kws
        scored: List[Tuple[float, str]] = []
        for s in donor_sents:
            kws = set(self._top_keywords(s, k=12))
            cover = len(missing & kws)
            if cover <= 0:
                continue
            penalty = 0.5 * math.log(1 + max(0, len(s) - 180) / 60.0)
            score = float(cover) - penalty
            if score > 0:
                scored.append((score, s.strip()))
        scored.sort(key=lambda x: x[0], reverse=True)

        take = [s for _, s in scored[: max(0, self.cfg.donor_sent_limit)]]
        if not take:
            reasons.append("donor_offers_no_new_goal_keywords")
            return cand, reasons

        if self.cfg.merge_strategy == "replace_worst" and cand_sents:
            cand_scored: List[Tuple[int, str, int]] = []  # (score, sent, idx)
            for i, s in enumerate(cand_sents):
                kws = set(self._top_keywords(s, k=12))
                cand_scored.append((len(kws & set(goal_kws)), s, i))
            cand_scored.sort(key=lambda x: x[0])  # weakest first
            out = cand_sents[:]
            for j, donor_sent in enumerate(take):
                if j < len(cand_scored):
                    idx = cand_scored[j][2]
                    out[idx] = donor_sent
            merged = " ".join(out)
            reasons.append("replaced_worst_candidate_sentences_with_donor")
        else:
            merged = (" ".join(cand_sents + take)).strip()
            reasons.append("appended_top_donor_sentences")

        merged = self._dedupe_by_signature(merged)
        return merged, reasons

    # ---- LLM prompt ----
    def _build_prompt(self, goal: str, cand: str, donor: str) -> str:
        return (
            f"{_ADAPT_SYSTEM_PROMPT}\n\n"
            f"Goal:\n{goal}\n\n"
            f"Donor:\n{donor[:1800]}\n\n"
            f"Draft:\n{cand}\n\n"
            f"Return only the improved text."
        )

    # ---- utilities ----
    def _finalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "").strip())
        sents = self._split_sentences(text)
        # clamp sentence count
        if len(sents) > self.cfg.max_sentences:
            sents = sents[: self.cfg.max_sentences]
        text = " ".join(sents)
        if len(text) > self.cfg.max_chars:
            text = text[: self.cfg.max_chars].rstrip()
            text = re.sub(r"\s+\S*$", "", text)
        # ensure final punctuation
        text = re.sub(r"([a-zA-Z0-9])\s*$", r"\1.", text)
        return text

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _tokens(self, text: str) -> List[str]:
        text = (text or "").lower()
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\[(?:\d+|[^\]]+)\]", " ", text)
        return re.findall(r"[a-z][a-z0-9_-]+", text)

    def _top_keywords(self, text: str, k: int = 12) -> List[str]:
        toks = [t for t in self._tokens(text) if len(t) >= self.cfg.min_kw_len and t not in _STOPWORDS]
        freq: Dict[str, int] = {}
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        return [w for w, _ in ranked[: max(1, k)]]

    def _coverage(self, goal: str, text: str) -> float:
        kws = set(self._top_keywords(goal, k=self.cfg.top_goal_kws))
        if not kws:
            return 0.0
        present = set(self._top_keywords(text, k=self.cfg.top_goal_kws))
        return float(len(kws & present)) / float(len(kws))

    def _dedupe_by_signature(self, text: str) -> str:
        sents = self._split_sentences(text)
        seen = set()
        out = []
        for s in sents:
            sig = re.sub(r"\W+", "", s.lower())
            if sig in seen:
                continue
            seen.add(sig)
            out.append(s)
        return " ".join(out)

    def _log(self, event: str, **fields):
        try:
            payload = {k: (str(v)[:240] if isinstance(v, str) else v) for k, v in (fields or {}).items()}
            # prefer your structured logger if present
            log = getattr(self.logger, "log", None)
            if callable(log):
                log(event, payload)
            else:
                self.logger.info(f"{event}: {payload}")
        except Exception:
            pass


# Minimal stopword list (kept short intentionally)
_STOPWORDS: set[str] = {
    "the","a","an","and","or","but","if","then","else","of","for","in","on","to","with","by","from","at","as","is","are","was","were","be","been","being","this","that","these","those","it","its","we","our","you","your","their","they","he","she","his","her","them","can","may","might","should","would","could","will","do","did","done","than","such","into","about","across","via","per","over","under","between","within","without","using","use","used","based","also","both","most","more","less","many","much","some","any","each","other","which","while","however","thus","therefore","hence","where","when","because","so","not",
}
