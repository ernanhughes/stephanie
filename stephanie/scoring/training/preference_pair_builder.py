# stephanie/scoring/mrq/preference_pair_builder.py

from collections import defaultdict
from typing import Dict, List, Iterable, Optional, Tuple
from stephanie.models.reasoning_sample import ReasoningSampleORM

def _iter_dimension_scores(sample: ReasoningSampleORM) -> Iterable[Tuple[str, float]]:
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

class PreferencePairBuilder:
    def __init__(self, memory, logger=None):
        self.memory = memory
        self.logger = logger

    def get_training_pairs_by_dimension(
        self,
        limit: int = 1000,
        dim: Optional[List[str]] = None,
        target_type: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, object]]]:
        results: Dict[str, List[Dict[str, object]]] = defaultdict(list)

        try:
            fetch_n = max(int(limit) * 4, 500)

            # ✅ Don’t call get_by_target_type with None
            if target_type:
                samples: List[ReasoningSampleORM] = self.memory.reasoning_samples.get_by_target_type(
                    target_type, limit=fetch_n
                )
            else:
                samples = self.memory.reasoning_samples.get_all(limit=fetch_n)

            by_dim: Dict[str, List[Tuple[float, str, str]]] = defaultdict(list)
            for s in samples or []:
                text = (getattr(s, "scorable_text", "") or "").strip()
                if not text:
                    continue
                title = getattr(s, "goal_text", "") or ""
                for dname, score in _iter_dimension_scores(s):
                    if dim and dname not in dim:
                        continue
                    by_dim[dname].append((score, text, title))

            remaining = int(limit)
            for dname, rows in by_dim.items():
                if not rows or remaining <= 0:
                    continue
                rows.sort(key=lambda r: r[0])                # asc
                k = max(1, min(len(rows) // 10, remaining))  # ~decile
                lows  = rows[:k]
                highs = rows[-k:][::-1]
                for (s_hi, text_hi, title_hi), (s_lo, text_lo, title_lo) in zip(highs, lows):
                    if not text_hi or text_hi == text_lo:
                        continue
                    results[dname].append({
                        "title": title_hi or title_lo or dname,
                        "output_a": text_hi,
                        "output_b": text_lo,
                        "value_a": float(s_hi),
                        "value_b": float(s_lo),
                    })
                    remaining -= 1
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break

        except Exception as e:
            if self.logger:
                self.logger.log("PreferencePairBuilderError", {"error": str(e)})
            # fall through with whatever we accumulated (likely empty)

        return dict(results)  # ✅ always a dict
