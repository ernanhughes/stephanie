# stephanie/scoring/mrq/preference_pair_builder.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

from stephanie.scoring.scorable import ScorableType

<<<<<<< HEAD
=======
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
>>>>>>> main

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

def _clamp01_or_100(v: float) -> float:
    # Accept 0..1 or 0..100; keep original scale (don’t normalize here).
    # This is just a sanity clamp to drop wild values.
    if 0.0 <= v <= 1.0:
        return v
    if 0.0 <= v <= 100.0:
        return v
    # Outside expected ranges → mark invalid by returning NaN
    return float("nan")


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
<<<<<<< HEAD
            if _nonempty(title) and _nonempty(a_txt) and _nonempty(b_txt) and a_val is not None and b_val is not None:
                a_val = _clamp01_or_100(a_val)
                b_val = _clamp01_or_100(b_val)
                if math.isfinite(a_val) and math.isfinite(b_val):
                    out.append({
                        "title": title,
                        "output_a": a_txt,
                        "output_b": b_txt,
                        "value_a": a_val,
                        "value_b": b_val,
                    })
                    kept += 1
                    continue
            dropped += 1

        if self.logger:
            self.logger.log("PreferencePairsFilteredPairwise", {
                "kept": kept, "dropped": dropped
            })
        return out

    def _filter_singletons(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept, dropped = 0, 0
        out: List[Dict[str, Any]] = []
        for s in items:
            # Accept several schema spellings
            title = (s.get("goal_text") or s.get("title") or "").strip()
            text  = (s.get("scorable_text") or s.get("output") or "").strip()
            score = _finite_score(s.get("target_score", s.get("score")))
            if _nonempty(title) and _nonempty(text) and score is not None:
                score = _clamp01_or_100(score)
                if math.isfinite(score):
                    out.append({
                        "goal_text": title,
                        "scorable_text": text,
                        "score": score,
                    })
                    kept += 1
                    continue
            dropped += 1

        if self.logger:
            self.logger.log("PreferencePairsFilteredSingletons", {
                "kept": kept, "dropped": dropped
            })
        return out

=======

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

>>>>>>> main
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
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch raw samples for a dimension, then:
          - ensure non-empty text fields,
          - ensure finite scores within expected ranges,
          - drop & log invalid rows.

        Returns:
            {dimension: [pair_dicts or singleton_dicts]}
        """
        try:
            samples = self.memory.reasoning_samples.get_eval_pairs_by_dimension(
                target_type=target_type, dimension=dimension, limit=limit
            )
            # sanitize payloads
            cleaned: list[dict] = []
            for d in samples.get(dimension, []):
                c = _clean_pair_dict(d)
                if c is not None:
                    cleaned.append(c)
                else:
                    if self.logger: self.logger.log("PrefPairInvalid", {"dimension": dimension})
            return {dimension: cleaned}
        except Exception as e:
            if self.logger:
                self.logger.log("PrefPairFetchError", {"dimension": dimension, "error": str(e)})
            return {dimension: []}


# stephanie/scoring/mrq/preference_pair_builder.py

def _is_valid_text(x: str | None) -> bool:
    return bool(x and isinstance(x, str) and x.strip())

def _is_valid_score(v) -> bool:
    try:
        f = float(v)
        return math.isfinite(f)
    except Exception:
        return False

def _clean_pair_dict(s: dict) -> dict | None:
<<<<<<< HEAD
    # Accept singleton or pairwise shapes; keep only if all required fields are valid
=======
>>>>>>> main
    title = (s.get("goal_text") or s.get("title") or "").strip()
    if not _is_valid_text(title):
        return None

    # singleton
    if "output" in s and ("score" in s or "target_score" in s):
        out = (s.get("scorable_text") or s.get("output") or "").strip()
        val = s.get("target_score", s.get("score"))
        if _is_valid_text(out) and _is_valid_score(val):
<<<<<<< HEAD
            return {"title": title, "output": out, "score": float(val)}
=======
            v01 = _normalize01(val)
            return {"title": title, "output": out, "score": v01} if math.isfinite(v01) else None
>>>>>>> main
        return None

    # pairwise
    if all(k in s for k in ("output_a","output_b","value_a","value_b")):
        a_ok = _is_valid_text(s.get("output_a")) and _is_valid_score(s.get("value_a"))
        b_ok = _is_valid_text(s.get("output_b")) and _is_valid_score(s.get("value_b"))
        if a_ok or b_ok:
<<<<<<< HEAD
            return {
                "title": title,
                "output_a": s.get("output_a") if a_ok else None,
                "value_a": float(s["value_a"]) if a_ok else None,
                "output_b": s.get("output_b") if b_ok else None,
                "value_b": float(s["value_b"]) if b_ok else None,
=======
            va01 = _normalize01(s["value_a"]) if a_ok else float("nan")
            vb01 = _normalize01(s["value_b"]) if b_ok else float("nan")
            return {
                "title": title,
                "output_a": s.get("output_a") if (a_ok and math.isfinite(va01)) else None,
                "value_a": va01 if (a_ok and math.isfinite(va01)) else None,
                "output_b": s.get("output_b") if (b_ok and math.isfinite(vb01)) else None,
                "value_b": vb01 if (b_ok and math.isfinite(vb01)) else None,
>>>>>>> main
            }
        return None

    # explicit HRM/MRQ form
    if ("goal_text" in s and "scorable_text" in s and ("target_score" in s or "score" in s)):
        out = (s.get("scorable_text") or "").strip()
        val = s.get("target_score", s.get("score"))
        if _is_valid_text(out) and _is_valid_score(val):
<<<<<<< HEAD
            return {"title": title, "output": out, "score": float(val)}
=======
            v01 = _normalize01(val)
            return {"title": title, "output": out, "score": v01} if math.isfinite(v01) else None
>>>>>>> main
    return None
