"""
TinyRecursionDataLoader
-----------------------
Builds TinyRecursionModel training samples from reasoning_samples_view.
Guaranteed inputs: goal_text, scorable_text.
Optional: scores, attributes, rationale.
"""

from __future__ import annotations
import json
from typing import List, Dict, Any

from stephanie.scoring.scorable import ScorableType


class TinyRecursionDataLoader:
    PREFERRED_DIMS = ("ai_knowledge", "reasoning")      # tweak order if you like
    PREFERRED_SOURCES = ("knowledge_llm", "llm", "sicql", "mrq", "ebt", "svm")

    def __init__(self, memory, logger=None):
        """
        Args:
            memory: Stephanie memory interface (with reasoning_samples store)
            logger: Optional JSONLogger or equivalent
        """
        self.memory = memory
        self.logger = logger


    def fetch_samples(self, limit: int = 5000) -> List[Dict[str, Any]]:
        samples = []
        rows = self.memory.reasoning_samples.get_by_target_type(
            ScorableType.CONVERSATION_TURN,
            limit=limit,
        )
        for r in rows:
            r = r.to_dict()
            goal_text = (r.get("goal_text") or "").strip()
            scorable_text = (r.get("scorable_text") or "").strip()
            if not goal_text or not scorable_text:
                continue

            scores = self._get_scores(r)          # ← unified reader
            attributes = self._get_attributes(r)  # ← unified reader

            # choose best score + rationale from scores
            chosen = self._choose_score(scores)
            if not chosen:
                # no usable score → skip
                continue

            dim, src, score, rationale = chosen
            target = self._to_int_0_100(score)    # ← 0..100 int
            z = self._build_reflection_text_from_choice(dim, src, score, rationale, attributes)

            samples.append({
                "x": goal_text,
                "y": scorable_text,
                "z": z,
                "target": target,                 # 0..100 int for CE
                "goal_id": r.get("goal_id"),
                "evaluation_id": r.get("evaluation_id"),
                "scorable_id": r.get("scorable_id"),
                "source": r.get("source"),
                "model_name": r.get("model_name"),
            })

        if self.logger:
            self.logger.log("TinyRecursionDataLoaded", {"count": len(samples)})
        return samples

    # ---------- helpers ----------

    def _get_scores(self, r) -> list:
        v = r.get("scores")
        if v is None:
            v = r.get("scores_json")
        return self._safe_json(v)

    def _get_attributes(self, r) -> list:
        v = r.get("attributes")
        if v is None:
            v = r.get("attributes_json")
        return self._safe_json(v)

    def _choose_score(self, scores: list):
        """
        Pick best score using (dimension priority, then source priority, then numeric score).
        Returns tuple: (dimension, source, score, rationale) or None.
        """
        if not scores:
            return None

        # rank by (dim_priority, src_priority, score)
        def dim_rank(d):
            d = (d or "").lower()
            for i, want in enumerate(self.PREFERRED_DIMS):
                if d == want:
                    return i
            return len(self.PREFERRED_DIMS)

        def src_rank(s):
            s = (s or "").lower()
            for i, want in enumerate(self.PREFERRED_SOURCES):
                if s == want:
                    return i
            return len(self.PREFERRED_SOURCES)

        best = None
        best_key = None
        for s in scores:
            d = s.get("dimension")
            src = s.get("source")
            sc = s.get("score")
            if sc is None:
                continue
            try:
                scf = float(sc)
            except Exception:
                continue

            key = (dim_rank(d), src_rank(src), -scf)  # higher score → smaller key via negative
            if (best is None) or (key < best_key):
                best = (d, src, scf, (s.get("rationale") or "").strip())
                best_key = key

        return best

    def _to_int_0_100(self, val: float) -> int:
        """
        Accepts 0..100 or 0..1; auto-scales 0..1 → 0..100, clamps, returns int.
        """
        try:
            v = float(val)
            if 0.0 <= v <= 1.0:
                v *= 100.0
            return int(max(0.0, min(100.0, v)))
        except Exception:
            return None

    def _build_reflection_text_from_choice(self, dim, src, score, rationale, attributes) -> str:
        lines = []
        if rationale:
            lines.append(f"{dim} ({src}={score}): {rationale}")
        # add a light attributes summary (optional)
        useful = []
        for a in attributes[:8]:
            d = a.get("dimension")
            e = a.get("energy")
            u = a.get("uncertainty")
            if e is not None or u is not None:
                useful.append(f"{d}: energy={e}, unc={u}")
        if useful:
            lines += useful
        if not lines:
            return "Reflecting on goal-context alignment using selected score."
        return "\n".join(lines[:10])

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _safe_json(self, val) -> list:
        """Gracefully decode JSON arrays from view columns."""
        if not val:
            return []
        if isinstance(val, list):
            return val
        try:
            return json.loads(val)
        except Exception:
            return []

    def _extract_rationale(self, scores: list) -> str:
        """Pull rationale from highest LLM score if available."""
        if not scores:
            return ""
        llm_scores = [s for s in scores if s.get("source") == "knowledge_llm"]
        top = max(llm_scores or scores, key=lambda s: s.get("score", 0))
        return (top.get("rationale") or "").strip()

    def _build_reflection_text(
        self, scores: list, attributes: list, rationale: str
    ) -> str:
        """
        Construct reflection text from scores and attributes.
        If none are present, fall back to rationale or generic reflection.
        """
        lines = []
        for s in scores:
            dim = s.get("dimension")
            src = s.get("source")
            score = s.get("score")
            rat = s.get("rationale")
            if rat and src == "knowledge_llm":
                lines.append(f"{dim} ({src}={score}): {rat}")

        for a in attributes:
            dim = a.get("dimension")
            energy = a.get("energy")
            unc = a.get("uncertainty")
            if energy is not None or unc is not None:
                lines.append(f"{dim}: energy={energy}, unc={unc}")

        if not lines and rationale:
            return rationale

        if not lines:
            return "No explicit rationale provided. Reflecting on goal-context alignment."

        return "\n".join(lines[:10])

    def _select_target(self, scores: list, scorable_text: str) -> str:
        """Pick the best target reasoning text."""
        target = scorable_text.strip()
        if not scores:
            return target
        top = max(scores, key=lambda s: s.get("score", 0))
        return (top.get("rationale") or target).strip()

    def _derive_halt_target(self, scores: list, attributes: list) -> int:
        """
        Heuristic for halting:
        High confidence if avg(score) > 80 and uncertainty < 0.2.
        """
        if not scores and not attributes:
            return 0

        avg_score = sum(s.get("score", 0.0) for s in scores) / max(1, len(scores))
        avg_unc = sum((a.get("uncertainty") or 0.0) for a in attributes) / max(
            1, len(attributes)
        )

        # normalize 0–100 to 0–1
        avg_score = avg_score / 100.0
        return int(avg_score > 0.8 and avg_unc < 0.2)
