# stephanie/engine/self_validation.py
from __future__ import annotations

import hashlib
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


class SelfValidationEngine:
    """
    Validates reward/ranker preferences against an LLM judge (or other oracle).
    Returns normalized metrics expected by TrainingController:
        - confidence: [0,1]   (agreement rate, weighted if weights present)
        - coverage:   [0,1]   (#validated / #pairs given)
        - regret:     [0,∞)   (1 - confidence) by default
        - sample_size: int    (#validated)
    Also logs per-pair diagnostics (truncated) and supports:
        - sampling by rate and/or max cap
        - tie handling (skip|half|strict)
        - weighted pairs
        - in-memory judgment cache with symmetric keys
    """

    def __init__(
        self,
        cfg,
        memory,
        logger,
        reward_model: callable,  # (goal, a, b, dimension=...) -> "a"|"b"|"tie"|numeric|dict
        llm_judge: callable,     # (goal, a, b, dimension=...) -> "a"|"b"|"tie"|numeric|dict
    ):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.reward_model = reward_model
        self.llm_judge = llm_judge

        # --- Config knobs (with sensible defaults) ---
        self.validation_sample_rate = float(getattr(cfg, "validation_sample_rate", 0.05))
        self.validation_max_pairs   = int(getattr(cfg, "validation_max_pairs", 64))
        self.validation_min_pairs   = int(getattr(cfg, "validation_min_pairs", 1))
        self.tie_mode               = str(getattr(cfg, "validation_tie_mode", "half")).lower()  # "half" | "skip" | "strict"
        self.truncate_chars         = int(getattr(cfg, "validation_truncate_chars", 300))
        self.random_seed            = getattr(cfg, "validation_seed", None)

        # Caching
        self.use_cache              = bool(getattr(cfg, "validation_use_cache", True))
        self.cache_size             = int(getattr(cfg, "validation_cache_size", 10000))
        self._cache: Dict[str, int] = {}  # key -> pref (+1 a, -1 b, 0 tie)
        self._cache_order: List[str] = []

        if self.random_seed is not None:
            random.seed(self.random_seed)

    # -------------------- Public API --------------------
    def validate_batch(self, goal: str, pairs: List[dict], dimension: str | None = None) -> Dict:
        """
        Args:
            goal: textual goal (query)
            pairs: list of pair dicts; supports any of these shapes:
              - {"text_a": str, "text_b": str, "weight": float?}
              - {"a": str, "b": str, "weight": float?}
              - {"doc_a": str, "doc_b": str, "weight": float?}
              - {"pos": str, "neg": str, "weight": float?}
              - {"query": str, "pos": str, "neg": str}  # query is ignored here; 'goal' param is used
            dimension: optional evaluation dimension (passed to model/judge if they accept it)

        Returns:
            dict with keys: confidence, coverage, regret, sample_size, validated, matches, logs
        """
        if not pairs:
            return {
                "confidence": 0.0,
                "coverage": 0.0,
                "regret": 1.0,
                "sample_size": 0,
                "validated": 0,
                "matches": 0.0,
                "logs": [],
            }

        # 1) Sample pairs
        pool = self._sample_pairs(pairs)

        validated = 0
        weighted_matches = 0.0
        total_weight = 0.0
        logs: List[Dict[str, Any]] = []

        for pair in pool:
            a, b, w = self._extract_pair(pair)
            if not a or not b:
                continue

            # a) model preference
            model_pref = self._pref_int(self._safe_call(self.reward_model, goal, a, b, dimension))

            # b) oracle (LLM) preference (with cache)
            llm_pref = self._judge_with_cache(goal, a, b, dimension)

            # c) agreement score in [0..1]
            match_score = self._match_score(model_pref, llm_pref)

            validated += 1
            total_weight += float(w)
            weighted_matches += float(w) * match_score

            # d) per-pair log
            entry = {
                "goal": goal,
                "dimension": dimension,
                "model_pref": self._pref_label(model_pref),
                "llm_pref": self._pref_label(llm_pref),
                "match_score": match_score,
                "weight": w,
                "text_a": (a[: self.truncate_chars] if isinstance(a, str) else str(a)) if a else "",
                "text_b": (b[: self.truncate_chars] if isinstance(b, str) else str(b)) if b else "",
            }
            self._log("SelfValidationResult", entry)
            logs.append(entry)

        # 2) Aggregate metrics
        if validated == 0 or total_weight == 0.0:
            confidence = 0.0
        else:
            confidence = max(0.0, min(1.0, weighted_matches / total_weight))

        coverage = validated / max(1, len(pairs))
        regret = 1.0 - confidence

        summary = {
            "confidence": round(confidence, 6),
            "coverage": round(coverage, 6),
            "regret": round(regret, 6),
            "sample_size": validated,
            "validated": validated,
            "matches": weighted_matches,  # weighted sum; keep raw for debugging
            "logs": logs,
        }

        self._log("SelfValidationSummary", {
            "goal": goal,
            "dimension": dimension,
            "confidence": summary["confidence"],
            "coverage": summary["coverage"],
            "regret": summary["regret"],
            "sample_size": validated,
        })

        # 3) Optional persistence
        try:
            if self.memory and hasattr(self.memory, "save"):
                self.memory.save("self_validation", {
                    "goal": goal,
                    "dimension": dimension,
                    "confidence": summary["confidence"],
                    "coverage": summary["coverage"],
                    "regret": summary["regret"],
                    "sample_size": validated,
                    "ts": time.time(),
                    # Avoid dumping all logs to persistent store unless you want that.
                    # "logs": logs,
                })
        except Exception as e:
            self._log("SelfValidationPersistError", {"error": str(e)})

        return summary

    # -------------------- Internals --------------------
    def _log(self, event: str, payload: Dict[str, Any]):
        if not self.logger:
            return
        try:
            # Your logger typically uses logger.log(event, {...})
            self.logger.log(event, payload)
        except TypeError:
            # Fallback if your logger expects 'extra='
            self.logger.log(event, extra=payload)

    def _sample_pairs(self, pairs: List[dict]) -> List[dict]:
        """Apply sampling rate and hard cap; ensure at least min_pairs if available."""
        # Bernoulli sampling
        if 0.0 < self.validation_sample_rate < 1.0:
            chosen = [p for p in pairs if random.random() < self.validation_sample_rate]
        else:
            chosen = list(pairs)

        # Ensure non-empty if we have data
        if not chosen and pairs:
            chosen = [random.choice(pairs)]

        # Hard cap
        if len(chosen) > self.validation_max_pairs:
            chosen = random.sample(chosen, self.validation_max_pairs)

        # Respect min_pairs (if we can)
        if len(chosen) < self.validation_min_pairs and len(pairs) >= self.validation_min_pairs:
            # top up randomly
            need = self.validation_min_pairs - len(chosen)
            remainder = [x for x in pairs if x not in chosen]
            chosen.extend(random.sample(remainder, min(need, len(remainder))))

        return chosen

    @staticmethod
    def _extract_pair(pair: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], float]:
        """
        Try multiple key schemes; return (a, b, weight).
        Weight defaults to 1.0 if not provided or invalid.
        """
        # Various possible key layouts
        a = pair.get("text_a") or pair.get("a") or pair.get("doc_a") or pair.get("pos")
        b = pair.get("text_b") or pair.get("b") or pair.get("doc_b") or pair.get("neg")
        # Some producers might place texts under "left"/"right"
        a = a or pair.get("left")
        b = b or pair.get("right")

        w = pair.get("weight", 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0

        return a, b, w

    @staticmethod
    def _pref_label(pref_int: int) -> str:
        return "a" if pref_int == 1 else ("b" if pref_int == -1 else "tie")

    def _match_score(self, model_pref: int, judge_pref: int) -> float:
        """
        Returns a match score in [0..1] considering tie handling:
          - "strict": ties must match exactly to count (1 if both tie else 0)
          - "half":   if either is a tie, count as 0.5; equal non-ties count 1
          - "skip":   if either is tie, count as 0 (but still contributes to coverage)
        """
        if model_pref == judge_pref:
            return 1.0
        if model_pref == 0 or judge_pref == 0:
            if self.tie_mode == "half":
                return 0.5
            elif self.tie_mode == "skip":
                return 0.0
            else:  # strict
                return 0.0
        return 0.0

    def _safe_call(self, fn: callable, goal: str, a: str, b: str, dimension: Optional[str]) -> Any:
        """
        Call model/judge with graceful fallback on signature differences.
        Accepts several return shapes; normalization happens in _pref_int.
        """
        try:
            return fn(goal, a, b, dimension=dimension)
        except TypeError:
            # Older signature (no dimension kw)
            return fn(goal, a, b)

    # --- Judgment cache (order-invariant) ---
    def _judge_with_cache(self, goal: str, a: str, b: str, dimension: Optional[str]) -> int:
        if not self.use_cache:
            return self._pref_int(self._safe_call(self.llm_judge, goal, a, b, dimension))

        key, flipped = self._cache_key(goal, a, b, dimension)
        if key in self._cache:
            pref = self._cache[key]
            return -pref if flipped else pref

        raw = self._safe_call(self.llm_judge, goal, a, b, dimension)
        pref = self._pref_int(raw)
        self._cache_put(key, pref)
        return -pref if flipped else pref

    @staticmethod
    def _canon_pair(a: str, b: str) -> Tuple[str, str, bool]:
        """
        Canonicalize (a,b) to be order-invariant:
        sort by hash; return (a_canon, b_canon, flipped)
        where flipped=True if the canonical order swapped original (a,b).
        """
        ha = hashlib.sha1((a or "").encode("utf-8")).hexdigest()
        hb = hashlib.sha1((b or "").encode("utf-8")).hexdigest()
        if ha <= hb:
            return a, b, False
        else:
            return b, a, True

    def _cache_key(self, goal: str, a: str, b: str, dimension: Optional[str]) -> Tuple[str, bool]:
        a_c, b_c, flipped = self._canon_pair(a, b)
        base = f"{goal}||{dimension or ''}||{a_c}||{b_c}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest(), flipped

    def _cache_put(self, key: str, pref: int):
        self._cache[key] = pref
        self._cache_order.append(key)
        if len(self._cache_order) > self.cache_size:
            # simple FIFO eviction (good enough for a validation cache)
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    # --- Preference normalization ---
    @staticmethod
    def _pref_int(x: Any) -> int:
        """
        Normalize various outputs to {-1,0,1} meaning {b, tie, a}.
        Accepts:
          - strings: "a"/"A"/"left"/"pos"/"first" → +1
                     "b"/"B"/"right"/"neg"/"second" → -1
                     "tie"/"equal" → 0
          - numbers: >0 → +1; <0 → -1; 0 → 0
          - dicts:   {'pref':'a'} or {'a':score_a,'b':score_b} or {'score_a':..,'score_b':..}
          - tuples:  (score_a, score_b)
        """
        if x is None:
            return 0

        # str
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"a", "left", "pos", "first"}:
                return 1
            if s in {"b", "right", "neg", "second"}:
                return -1
            if s in {"tie", "equal", "both", "none"}:
                return 0
            # permissive: "a "* or " b"
            if s.startswith("a"):
                return 1
            if s.startswith("b"):
                return -1
            return 0

        # number
        if isinstance(x, (int, float)):
            if x > 0:
                return 1
            if x < 0:
                return -1
            return 0

        # dict: look for explicit pref first
        if isinstance(x, dict):
            pref = x.get("pref") or x.get("preference") or x.get("winner")
            if pref is not None:
                return SelfValidationEngine._pref_int(pref)

            # try scores
            a_score = x.get("a") if isinstance(x.get("a"), (int, float)) else x.get("score_a")
            b_score = x.get("b") if isinstance(x.get("b"), (int, float)) else x.get("score_b")
            if isinstance(a_score, (int, float)) and isinstance(b_score, (int, float)):
                diff = float(a_score) - float(b_score)
                return 1 if diff > 0 else (-1 if diff < 0 else 0)

        # tuple of scores
        if isinstance(x, (tuple, list)) and len(x) >= 2:
            try:
                a_score = float(x[0])
                b_score = float(x[1])
                diff = a_score - b_score
                return 1 if diff > 0 else (-1 if diff < 0 else 0)
            except Exception:
                return 0

        # fallback
        return 0
