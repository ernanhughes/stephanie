# stephanie/scoring/dataloader/tiny_recursion_dataloader.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import json

from stephanie.scoring.scorable import ScorableType

# optional tqdm
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


class TinyRecursionDataLoader:
    """
    Builds TinyRecursionModel training samples from reasoning_samples_view.

    Each returned sample has:
      {
        "x": goal_text (str),
        "y": scorable_text (str),
        "z": reflection (str),
        "target": int in [0, 100],
        "halt_target": float in {0.0, 1.0},
        ... plus metadata (goal_id, evaluation_id, scorable_id, source, model_name)
      }
    """

    # defaults prioritize knowledge judgment first, reasoning second
    DEFAULT_PREFERRED_DIMS = ("knowledge_value", "reasoning")
    DEFAULT_PREFERRED_SOURCES = ("knowledge_llm", "llm", "sicql", "mrq", "ebt", "svm")

    def __init__(
        self,
        memory,
        logger=None,
        *,
        preferred_dims: Tuple[str, ...] = DEFAULT_PREFERRED_DIMS,
        preferred_sources: Tuple[str, ...] = DEFAULT_PREFERRED_SOURCES,
        use_calibrated_score: bool = False,
        show_progress: bool = True,
        label_hist_bucket: int = 10,
        min_score: Optional[float] = None,   # drop if chosen target < min_score (after 0..100 scaling)
        drop_missing_reflection: bool = False,  # if True, drop when we cannot build any reflection
    ):
        self.memory = memory
        self.logger = logger
        self.PREFERRED_DIMS = tuple(preferred_dims or self.DEFAULT_PREFERRED_DIMS)
        self.PREFERRED_SOURCES = tuple(preferred_sources or self.DEFAULT_PREFERRED_SOURCES)
        self.use_calibrated_score = bool(use_calibrated_score)
        self.show_progress = bool(show_progress)
        self.label_hist_bucket = int(label_hist_bucket)
        self.min_score = float(min_score) if min_score is not None else None
        self.drop_missing_reflection = bool(drop_missing_reflection)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fetch_samples(self, limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Pull structured reasoning samples from reasoning_samples_view
        for CONVERSATION_TURN target type and convert them into trainer samples.
        """
        rows = self.memory.reasoning_samples.get_by_target_type(
            ScorableType.CONVERSATION_TURN,
            limit=int(limit),
        )

        samples: List[Dict[str, Any]] = []
        kept = 0
        dropped = 0
        label_counts = Counter()

        use_tqdm = bool(self.show_progress and tqdm is not None)
        pbar = tqdm(rows, desc="Fetching TinyRecursion samples", unit="row") if use_tqdm else None
        iterator = pbar if pbar is not None else rows

        for r in iterator:
            try:
                rec = r.to_dict() if hasattr(r, "to_dict") else dict(r)
                goal_text = (rec.get("goal_text") or "").strip()
                scorable_text = (rec.get("scorable_text") or "").strip()

                if not goal_text or not scorable_text:
                    dropped += 1
                    self._log_drop("missing_text", rec)
                    continue

                scores = self._get_scores(rec)
                attributes = self._get_attributes(rec)

                # choose best score as target (0..100 int)
                choose = self._choose_score(scores)
                if not choose:
                    dropped += 1
                    self._log_drop("no_usable_score", rec)
                    continue

                dim, src, score_val, rationale = choose
                target = self._to_int_0_100(score_val)
                if target is None:
                    dropped += 1
                    self._log_drop("target_cast_failed", rec)
                    continue

                if self.min_score is not None and target < self.min_score:
                    dropped += 1
                    self._log_drop("below_min_score", rec)
                    continue

                # reflection text z
                z_text = self._build_reflection_text_from_choice(dim, src, score_val, rationale, attributes)
                if not z_text and self.drop_missing_reflection:
                    dropped += 1
                    self._log_drop("no_reflection", rec)
                    continue

                halt_target = self._derive_halt_target(scores, attributes)

                sample = {
                    "x": goal_text,
                    "y": scorable_text,
                    "z": z_text or "Reflection: assess alignment.",
                    "target": target,
                    "halt_target": float(halt_target),
                    # meta
                    "goal_id": rec.get("goal_id"),
                    "evaluation_id": rec.get("evaluation_id"),
                    "scorable_id": rec.get("scorable_id"),
                    "source": rec.get("source"),
                    "model_name": rec.get("model_name"),
                }
                samples.append(sample)
                kept += 1
                label_counts[target] += 1

                if pbar is not None:
                    pbar.set_postfix(kept=kept, drop=dropped)

            except Exception as e:
                dropped += 1
                if self.logger:
                    self.logger.log("TinyRecursionDataLoaderError", {"error": str(e)})

        if pbar is not None:
            pbar.close()

        # histograms + summary
        self._log_label_histogram(kept, dropped, label_counts)
        if self.logger:
            self.logger.log("TinyRecursionDataLoaded", {"count": len(samples), "kept": kept, "dropped": dropped})

        return samples

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _get_scores(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        v = rec.get("scores")
        if not v:
            v = rec.get("scores_json")
        return self._safe_json_list(v)

    def _get_attributes(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        v = rec.get("attributes")
        if not v:
            v = rec.get("attributes_json")
        return self._safe_json_list(v)

    def _safe_json_list(self, val) -> List[Dict[str, Any]]:
        if isinstance(val, list):
            return val
        if not val:
            return []
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def _choose_score(self, scores: List[Dict[str, Any]]) -> Optional[Tuple[str, str, float, str]]:
        """
        Pick best score by (dimension preference, source preference, score desc).
        Returns tuple: (dimension, source, score_value_float, rationale_str)
        """
        if not scores:
            return None

        dim_pref = [d.lower() for d in self.PREFERRED_DIMS]
        src_pref = [s.lower() for s in self.PREFERRED_SOURCES]

        def dim_rank(d: Optional[str]) -> int:
            d = (d or "").lower()
            try:
                return dim_pref.index(d)  # exact match priority
            except ValueError:
                # allow partial match as a fallback
                for i, pref in enumerate(dim_pref):
                    if pref in d and pref:
                        return i + len(dim_pref)
                return 1_000_000  # deprioritize

        def src_rank(s: Optional[str]) -> int:
            s = (s or "").lower()
            try:
                return src_pref.index(s)
            except ValueError:
                for i, pref in enumerate(src_pref):
                    if pref in s and pref:
                        return i + len(src_pref)
                return 1_000_000

        best = None
        best_key = None
        for s in scores:
            raw_score = s.get("calibrated_score") if self.use_calibrated_score and s.get("calibrated_score") is not None else s.get("score")
            if raw_score is None:
                continue
            try:
                val = float(raw_score)
            except Exception:
                continue

            # normalize if in 0..1 range
            if 0.0 <= val <= 1.0:
                val *= 100.0

            key = (dim_rank(s.get("dimension")), src_rank(s.get("source")), -val)
            if best is None or key < best_key:
                best = (s.get("dimension"), s.get("source"), val, (s.get("rationale") or "").strip())
                best_key = key

        return best

    def _to_int_0_100(self, val: float) -> Optional[int]:
        try:
            v = float(val)
            if 0.0 <= v <= 1.0:
                v *= 100.0
            return int(max(0, min(100, round(v))))
        except Exception:
            return None

    def _build_reflection_text_from_choice(
        self, dim: Optional[str], src: Optional[str], score: float, rationale: str, attributes: List[Dict[str, Any]]
    ) -> str:
        lines = []
        if rationale:
            lines.append(f"{dim or 'score'} ({src or 'source'}={int(round(score))}): {rationale}")

        # Attach a few attribute diagnostics, if any
        for a in (attributes or [])[:8]:
            d = a.get("dimension")
            e = a.get("energy")
            u = a.get("uncertainty")
            if e is not None or u is not None:
                lines.append(f"{d}: energy={e}, unc={u}")

        return "\n".join(lines[:10])

    def _derive_halt_target(self, scores: List[Dict[str, Any]], attributes: List[Dict[str, Any]]) -> float:
        """
        Heuristic: halt=1.0 if avg(score)>80 and avg_uncertainty<0.2, else 0.0
        Accepts scores either in 0..100 or 0..1; auto-normalizes.
        """
        if not scores and not attributes:
            return 0.0

        # avg score
        sv = []
        for s in scores:
            v = s.get("score")
            if v is None:
                continue
            try:
                vf = float(v)
                if 0.0 <= vf <= 1.0:
                    vf *= 100.0
                sv.append(vf)
            except Exception:
                continue
        avg_score = (sum(sv) / max(1, len(sv))) if sv else 0.0

        # avg uncertainty
        uv = []
        for a in attributes or []:
            u = a.get("uncertainty")
            if u is None:
                continue
            try:
                uv.append(float(u))
            except Exception:
                continue
        avg_unc = (sum(uv) / max(1, len(uv))) if uv else 0.0

        return 1.0 if (avg_score > 80.0 and avg_unc < 0.2) else 0.0

    # ---------------------------------------------------------------------
    # Logging helpers
    # ---------------------------------------------------------------------
    def _log_drop(self, reason: str, rec: Dict[str, Any]):
        if self.logger:
            self.logger.log("TinyRecursionSampleDropped", {
                "reason": reason,
                "evaluation_id": rec.get("evaluation_id"),
                "scorable_id": rec.get("scorable_id"),
                "model_name": rec.get("model_name"),
                "source": rec.get("source"),
            })

    def _log_label_histogram(self, kept: int, dropped: int, label_counts: Counter):
        if not self.logger:
            return
        exact = {int(k): int(v) for k, v in sorted(label_counts.items())}
        bucketed = self._bucketize_counts(label_counts, self.label_hist_bucket)
        self.logger.log("TinyRecursionLabelHistogram", {
            "kept": int(kept),
            "dropped": int(dropped),
            "exact": exact,
            "bucket_size": int(self.label_hist_bucket),
            "bucketed": bucketed,
        })

    def _bucketize_counts(self, counts: Counter, bucket: int) -> dict:
        if bucket <= 1:
            return {str(k): int(v) for k, v in sorted(counts.items())}
        buckets = {}
        for label, c in counts.items():
            try:
                l = int(label)
            except Exception:
                continue
            start = (l // bucket) * bucket
            end = min(100, start + bucket - 1)
            key = f"{start}-{end}"
            buckets[key] = buckets.get(key, 0) + int(c)
        # make ranges dense for nicer plots
        start = 0
        while start <= 100:
            end = min(100, start + bucket - 1)
            key = f"{start}-{end}"
            buckets.setdefault(key, 0)
            start += bucket
        return dict(sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0])))
