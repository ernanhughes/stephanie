# stephanie/scoring/metrics/feature/rubric_judge_group_feature.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from stephanie.scoring.metrics.feature.base_group_feature import \
    BaseGroupFeature
from stephanie.tools.rubric_judge_tool import RubricJudgeTool

log = logging.getLogger(__name__)


class RubricJudgeGroupFeature(BaseGroupFeature):
    """
    Group-wise LLM-as-a-judge feature.

    Intended to run AFTER PairRM ranking:

      1. Group rows by `group_key` (e.g. seed_id).
      2. For each group, select rows where `rank_filter_key` <= `max_rank_for_judging`
         (e.g. top 2–3 by PairRM rank).
      3. Use RubricJudgeTool to score each selected candidate against a shared context.
      4. Write:
           - judge_overall
           - judge_<criterion>_score
           - judge_<criterion>_rationale
         into each row.

    Rows outside the top-K (or missing rank) get None for judge_* fields.
    """

    name = "rubric_judge_group"
    requires: List[str] = []

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        tool_cfg = cfg.get("tools", {}).get("rubric_judge", cfg)
        self.tool = RubricJudgeTool(tool_cfg, memory, container, logger)

        self.enabled: bool = bool(cfg.get("enabled", True))

        # Grouping / selection
        self.group_key: str = cfg.get("group_key", "group_id")
        self.candidate_text_column: str = cfg.get(
            "candidate_text_column", "blog_text"
        )
        self.context_text_column: Optional[str] = cfg.get(
            "context_text_column", "context_text"
        )
        self.candidate_id_column: Optional[str] = cfg.get(
            "candidate_id_column", "id"
        )

        # Use this key (e.g. "pairrm_rank") to select top-K for judging.
        self.rank_filter_key: str = cfg.get("rank_filter_key", "pairrm_rank")
        self.max_rank_for_judging: int = int(cfg.get("max_rank_for_judging", 2))

        # Output keys
        self.overall_key: str = cfg.get("overall_key", "judge_overall")
        self.prefix: str = cfg.get("prefix", "judge_")
        self.group_label_key: str = cfg.get("group_label_key", "judge_group")

        # Telemetry
        self._rows_in: int = 0
        self._rows_judged: int = 0
        self._groups_seen: int = 0
        self._groups_with_judgements: int = 0
        self._error: Optional[str] = None

    async def apply(
        self, rows: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not rows:
            return rows

        self._reset_telemetry(len(rows))

        # Group indices by group_key
        groups: Dict[Any, List[int]] = {}
        for idx, row in enumerate(rows):
            group_val = row.get(self.group_key)
            if group_val is None:
                group_val = f"__row_{idx}"
            groups.setdefault(group_val, []).append(idx)

        pipeline_context_text: str = context.get("judge_context", "") or ""

        for group_id, indices in groups.items():
            self._groups_seen += 1
            group_rows = [rows[i] for i in indices]

            # Shared context for this group
            context_text = self._resolve_context_text(
                group_rows, pipeline_context_text
            )

            # First: initialise judge_* keys so they always exist
            for row_index in indices:
                row = rows[row_index]
                row.setdefault(self.overall_key, None)
                row.setdefault(self.group_label_key, str(group_id))

            # Select top-K rows by rank_filter_key
            ranked_indices = [
                (idx, rows[idx])
                for idx in indices
                if self._valid_rank(rows[idx].get(self.rank_filter_key))
            ]

            if not ranked_indices:
                # No ranked rows in this group → nothing to judge
                continue

            ranked_indices.sort(key=lambda t: t[1][self.rank_filter_key])  # ascending
            top_k = ranked_indices[: self.max_rank_for_judging]

            any_judged = False

            for row_index, row in top_k:
                blog_text = row.get(self.candidate_text_column)
                if not isinstance(blog_text, str) or not blog_text.strip():
                    continue

                candidate_id = None
                if self.candidate_id_column is not None:
                    candidate_id = row.get(self.candidate_id_column)
                if candidate_id is None:
                    candidate_id = f"{group_id}:{row_index}"

                try:
                    result = self.tool.evaluate(
                        context_text=context_text,
                        candidate_text=blog_text,
                        candidate_id=str(candidate_id),
                    )
                except Exception as e:
                    self._error = f"{type(e).__name__}: {e}"
                    log.warning(
                        "[RubricJudgeGroupFeature] Judge failed for group=%r row_index=%r: %s",
                        group_id,
                        row_index,
                        self._error,
                    )
                    continue

                self._rows_judged += 1
                any_judged = True

                # Attach overall and per-criterion fields
                row[self.overall_key] = result.overall_score
                # Each criterion becomes judge_<name>_score and judge_<name>_rationale
                for name, crit in result.criteria.items():
                    row[f"{self.prefix}{name}_score"] = crit.score
                    row[f"{self.prefix}{name}_rationale"] = crit.rationale

            if any_judged:
                self._groups_with_judgements += 1

        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_context_text(
        self,
        group_rows: List[Dict[str, Any]],
        pipeline_context_text: str,
    ) -> str:
        """
        Priority:
          1. First non-empty row[context_text_column] if configured.
          2. pipeline-level judge_context.
          3. Empty string.
        """
        if self.context_text_column:
            for r in group_rows:
                ctx = r.get(self.context_text_column)
                if isinstance(ctx, str) and ctx.strip():
                    return ctx
        return pipeline_context_text

    def _valid_rank(self, value: Any) -> bool:
        """
        Is this a valid rank for judging? (non-None, int, >= 0)
        """
        try:
            if value is None:
                return False
            v = int(value)
            return v >= 0
        except Exception:
            return False

    def _reset_telemetry(self, rows_in: int) -> None:
        self._rows_in = rows_in
        self._rows_judged = 0
        self._groups_seen = 0
        self._groups_with_judgements = 0
        self._error = None

    def report(self) -> Dict[str, Any]:
        ok = self._error is None
        return {
            "feature": self.name,
            "ok": ok,
            "rows_in": self._rows_in,
            "rows_judged": self._rows_judged,
            "groups_seen": self._groups_seen,
            "groups_with_judgements": self._groups_with_judgements,
            "error": self._error,
            "summary": (
                f"rows_judged={self._rows_judged}/{self._rows_in}; "
                f"groups_with_judgements={self._groups_with_judgements}/"
                f"{self._groups_seen}"
            ),
        }
