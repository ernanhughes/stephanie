# stephanie/scoring/mrq/preference_pair_builder.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from stephanie.scoring.scorable import ScorableType


def _normalize01(v: float) -> float:
    """
    Accept 0..1 or 0..100 and return 0..1.
    If v > 1 but <= 100, treat as percent.
    Clamp to [0,1]; NaN/inf → None handled by caller.
    """
    try:
        f = float(v)
    except Exception:
        return float("nan")

    if not math.isfinite(f):
        return float("nan")

    if 0.0 <= f <= 1.0:
        return f
    if 0.0 <= f <= 100.0:
        return f / 100.0
    # out of range → NaN so it gets dropped
    return float("nan")

def _nonempty(s: Optional[str], min_len: int = 3) -> bool:
    return isinstance(s, str) and (s.strip() != "") and (len(s.strip()) >= min_len)

def _finite_score(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None

class PreferencePairBuilder:
    """
    Builds preference training pairs: (high-score sample, low-score sample) per dimension.

    Returns:
        {dimension: List[ dict ] } where each dict is EITHER:
          Pairwise:
            {
              "title": str,
              "output_a": str,  # better
              "output_b": str,  # worse
              "value_a": float,
              "value_b": float
            }
          or Singleton (for pipelines that consume singletons):
            {
              "goal_text": str,        # or "title"
              "scorable_text": str,    # or "output"
              "score": float           # or "target_score"
            }
    All returned rows are guaranteed to have non-empty text and finite scores.
    """

    def __init__(self, memory, logger=None):
        self.memory = memory
        self.logger = logger

    def _filter_pairwise(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept, dropped = 0, 0
        out: List[Dict[str, Any]] = []
        for s in items:
            title = (s.get("title") or s.get("goal_text") or "").strip()
            a_txt = (s.get("output_a") or "").strip()
            b_txt = (s.get("output_b") or "").strip()
            a_val = _finite_score(s.get("value_a"))
            b_val = _finite_score(s.get("value_b"))

            if _nonempty(title) and _nonempty(a_txt) and _nonempty(b_txt) and a_val is not None and b_val is not None:
                a01 = _normalize01(a_val)
                b01 = _normalize01(b_val)
                if math.isfinite(a01) and math.isfinite(b01):
                    out.append({
                        "title": title,
                        "output_a": a_txt,
                        "output_b": b_txt,
                        "value_a": a01,     # <<< normalized 0..1
                        "value_b": b01,     # <<< normalized 0..1
                    })
                    kept += 1
                    continue
            dropped += 1

        if self.logger:
            self.logger.log("PreferencePairsFilteredPairwise", {"kept": kept, "dropped": dropped})
        return out

    def _filter_singletons(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept, dropped = 0, 0
        out: List[Dict[str, Any]] = []
        for s in items:
            title = (s.get("goal_text") or s.get("title") or "").strip()
            text  = (s.get("scorable_text") or s.get("output") or "").strip()
            score = _finite_score(s.get("target_score", s.get("score")))
            if _nonempty(title) and _nonempty(text) and score is not None:
                s01 = _normalize01(score)
                if math.isfinite(s01):
                    out.append({
                        "goal_text": title,
                        "scorable_text": text,
                        "score": s01,       # <<< normalized 0..1
                    })
                    kept += 1
                    continue
            dropped += 1

        if self.logger:
            self.logger.log("PreferencePairsFilteredSingletons", {"kept": kept, "dropped": dropped})
        return out

    def _detect_and_filter(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        The upstream view may mix pairwise & singleton rows. Split, filter, rejoin.
        """
        pairs, singles = [], []
        for s in raw:
            if all(k in s for k in ("output_a", "output_b", "value_a", "value_b")):
                pairs.append(s)
            else:
                singles.append(s)
        pairs_f  = self._filter_pairwise(pairs)   if pairs   else []
        sing_f   = self._filter_singletons(singles) if singles else []
        return pairs_f + sing_f

    def get_training_pairs_by_dimension(
        self,
        dimension: str = "knowledge",
        target_type: str = ScorableType.CONVERSATION_TURN,
        limit: int = 1000,
        *,
        text_max: Optional[int] = None,
        fetch_multiplier: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch raw samples for a dimension, sanitize, enforce optional text-length
        constraints, and return up to `limit` rows.

        Length rule (when text_max is provided):
          - Drop if goal/title length > text_max
          - For singletons: drop if scorable/output length > text_max
          - For pairwise: drop if either output_a or output_b is missing or exceeds text_max

        We oversample upstream (limit * fetch_multiplier) when text_max is set
        to compensate for dropped rows, then slice to `limit`.
        """
        kept_len = 0
        dropped_len = 0

        try:
            # Oversample if we're going to filter by length,
            # so we still have a good chance of returning `limit`.
            fetch_n = int(limit * fetch_multiplier) if (text_max and limit) else limit

            samples = self.memory.reasoning_samples.get_eval_pairs_by_dimension(
                target_type=target_type, dimension=dimension, limit=fetch_n
            )

            # 1) Clean normalization/shape (0..1, valid fields)
            cleaned: list[dict] = []
            for d in samples.get(dimension, []):
                c = _clean_pair_dict(d)
                if c is not None:
                    cleaned.append(c)
                else:
                    if self.logger:
                        self.logger.log("PrefPairInvalid", {"dimension": dimension})

            # 2) Apply strict text-length filter if requested
            if text_max is not None and text_max > 0:
                filtered: list[dict] = []
                for row in cleaned:
                    title = (row.get("title") or "").strip()
                    if not _within_len(title, text_max):
                        dropped_len += 1
                        continue

                    if "output" in row:
                        # singleton
                        out = (row.get("output") or "").strip()
                        if _within_len(out, text_max):
                            filtered.append(row)
                            kept_len += 1
                        else:
                            dropped_len += 1
                    elif ("output_a" in row) or ("output_b" in row):
                        # pairwise: both must be present and within limit
                        a = row.get("output_a")
                        b = row.get("output_b")
                        if a is None or b is None:
                            dropped_len += 1
                            continue
                        a = a.strip()
                        b = b.strip()
                        if _within_len(a, text_max) and _within_len(b, text_max):
                            filtered.append(row)
                            kept_len += 1
                        else:
                            dropped_len += 1
                    else:
                        # Unknown shape—be conservative and drop
                        dropped_len += 1

                cleaned = filtered

            # 3) Trim to requested limit
            result = cleaned[: max(0, int(limit))]

            if self.logger:
                if text_max:
                    self.logger.log("PreferencePairsLengthFilter",
                                    {"dimension": dimension,
                                     "text_max": int(text_max),
                                     "kept": kept_len,
                                     "dropped": dropped_len,
                                     "returned": len(result),
                                     "requested_limit": int(limit),
                                     "fetched": int(fetch_n)})
                else:
                    self.logger.log("PreferencePairsNoLengthFilter",
                                    {"dimension": dimension,
                                     "returned": len(result),
                                     "requested_limit": int(limit)})

            return {dimension: result}

        except Exception as e:
            if self.logger:
                self.logger.log("PrefPairFetchError", {"dimension": dimension, "error": str(e)})
            return {dimension: []}


def _is_valid_text(x: str | None) -> bool:
    return bool(x and isinstance(x, str) and x.strip())

def _within_len(s: Optional[str], maxlen: int) -> bool:
    return isinstance(s, str) and (len(s) <= maxlen)

def _is_valid_score(v) -> bool:
    try:
        f = float(v)
        return math.isfinite(f)
    except Exception:
        return False

def _clean_pair_dict(s: dict) -> dict | None:
    title = (s.get("goal_text") or s.get("title") or "").strip()
    if not _is_valid_text(title):
        return None

    # singleton
    if "output" in s and ("score" in s or "target_score" in s):
        out = (s.get("scorable_text") or s.get("output") or "").strip()
        val = s.get("target_score", s.get("score"))
        if _is_valid_text(out) and _is_valid_score(val):
            v01 = _normalize01(val)
            return {"title": title, "output": out, "score": v01} if math.isfinite(v01) else None
        return None

    # pairwise
    if all(k in s for k in ("output_a","output_b","value_a","value_b")):
        a_ok = _is_valid_text(s.get("output_a")) and _is_valid_score(s.get("value_a"))
        b_ok = _is_valid_text(s.get("output_b")) and _is_valid_score(s.get("value_b"))
        if a_ok or b_ok:
            va01 = _normalize01(s["value_a"]) if a_ok else float("nan")
            vb01 = _normalize01(s["value_b"]) if b_ok else float("nan")
            return {
                "title": title,
                "output_a": s.get("output_a") if (a_ok and math.isfinite(va01)) else None,
                "value_a": va01 if (a_ok and math.isfinite(va01)) else None,
                "output_b": s.get("output_b") if (b_ok and math.isfinite(vb01)) else None,
                "value_b": vb01 if (b_ok and math.isfinite(vb01)) else None,
            }
        return None

    # explicit HRM/MRQ form
    if ("goal_text" in s and "scorable_text" in s and ("target_score" in s or "score" in s)):
        out = (s.get("scorable_text") or "").strip()
        val = s.get("target_score", s.get("score"))
        if _is_valid_text(out) and _is_valid_score(val):
            v01 = _normalize01(val)
            return {"title": title, "output": out, "score": v01} if math.isfinite(v01) else None
    return None
