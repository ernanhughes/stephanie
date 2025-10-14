# stephanie/memory/reasoning_sample_store.py
from __future__ import annotations
from typing import List
from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.reasoning_sample import ReasoningSampleORM
from typing import Dict, List, Iterable, Tuple, Optional
from collections import defaultdict


def _iter_dimension_scores(sample: ReasoningSampleORM) -> Iterable[Tuple[str, float]]:
    """
    Yield (dimension, score) pairs from ReasoningSampleORM.scores JSON.
    Each item in `scores` should look like: {"dimension": "...", "score": <number>, ...}.
    Skips non-numeric scores or missing dimensions.
    """
    items = getattr(sample, "scores", None) or []
    for it in items:
        dim = it.get("dimension")
        val = it.get("score")
        if not dim:
            continue
        try:
            yield dim, float(val)
        except Exception:
            continue

class ReasoningSampleStore(BaseSQLAlchemyStore):
    """
    Read-only store for reasoning_samples_view.
    Used by data loaders (TinyRecursion, SICQL, etc.)
    to fetch structured reasoning examples.
    """
    orm_model = ReasoningSampleORM
    default_order_by = ReasoningSampleORM.created_at.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "reasoning_samples"

    def get_all(self, limit: int = 1000) -> List[ReasoningSampleORM]:
        def op(s):
            return (
                s.query(ReasoningSampleORM)
                .order_by(ReasoningSampleORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)


    def get_by_target_type(self, target_type: str, limit: int = 100) -> List[ReasoningSampleORM]:
        def op(s):
            return (
                s.query(ReasoningSampleORM)
                .filter(ReasoningSampleORM.scorable_type == target_type)
                .order_by(ReasoningSampleORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def get_by_goal(self, goal_text: str, limit: int = 50) -> List[ReasoningSampleORM]:
        def op(s):
            return (
                s.query(ReasoningSampleORM)
                .filter(ReasoningSampleORM.goal_text.ilike(f"%{goal_text}%"))
                .limit(limit)
                .all()
            )
        return self._run(op)


    def get_training_pairs_by_dimension(
        self,
        goal: Optional[str] = None,
        limit: int = 100,
        dim: Optional[List[str]] = None,
        target_type: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, object]]]:
        """
        Build preference pairs (TOP vs BOTTOM) per (dimension × scorable_id),
        mirroring the old SQL:
          - Partition by (dimension, doc_id) and pick rank_high=1 and rank_low=1
          - Keep non-empty text
          - Return pairs grouped by dimension

        Args:
          goal: optional substring filter on goal_text (ILIKE)
          limit: total pairs to return (best-effort across all dimensions)
          dim: optional list of dimension names to include
          target_type: optional filter on scorable_type (e.g., "document")

        Returns:
          { dimension: [ {title, output_a, output_b, value_a, value_b}, ... ] }
        """
        # 1) Fetch a generous slice of recent samples
        fetch_n = max(limit * 20, 500)

        if goal and target_type:
            samples = self.get_by_target_type(target_type, limit=fetch_n)
            g = goal.lower()
            samples = [s for s in samples if (getattr(s, "goal_text", "") or "").lower().find(g) >= 0][:fetch_n]
        elif target_type:
            samples = self.get_by_target_type(target_type, limit=fetch_n)
        elif goal:
            samples = self.get_by_goal(goal, limit=fetch_n)
        else:
            samples = self.get_all(limit=fetch_n)

        # 2) Bucket by (dimension, scorable_id) collecting candidate rows with scores
        # We store: for each key, a list of tuples (score, text, title, sample_ref)
        by_key: Dict[Tuple[str, str], List[Tuple[float, str, str, ReasoningSampleORM]]] = defaultdict(list)

        for s in samples:
            text = (getattr(s, "scorable_text", None) or "").strip()
            title = getattr(s, "goal_text", None) or ""
            scorable_id = getattr(s, "scorable_id", None)
            if not scorable_id:
                continue  # must have an id to emulate doc_id partitioning

            for dname, score in _iter_dimension_scores(s):
                if dim and dname not in dim:
                    continue
                # Note: allow empty text for bottom side? Original SQL required non-empty for 'top' only.
                by_key[(dname, scorable_id)].append((score, text, title, s))

        # 3) For each (dimension, scorable_id), pick highest and lowest scored sample
        results_by_dimension: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        remaining = limit

        for (dname, _doc_id), rows in by_key.items():
            if not rows:
                continue

            # rank_high = 1 (max score), rank_low = 1 (min score)
            rows_sorted = sorted(rows, key=lambda r: r[0])
            low = rows_sorted[0]
            high = rows_sorted[-1]

            score_hi, text_hi, title_hi, _ = high
            score_lo, text_lo, title_lo, _ = low

            # Emulate original filter: top.text must be non-empty
            if not text_hi:
                continue

            # Build pair (top vs bottom). The original code allowed bottom to be empty;
            # we keep it as-is to mirror behavior.
            results_by_dimension[dname].append({
                "title": title_hi or title_lo or dname,
                "output_a": text_hi,
                "output_b": text_lo,
                "value_a": float(score_hi),
                "value_b": float(score_lo),
            })
            remaining -= 1
            if remaining <= 0:
                break

        return dict(results_by_dimension)
