from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional

from stephanie.scoring.metrics.feature.base_group_feature import \
    BaseGroupFeature
from stephanie.tools.pairrm_ranking_tool import PairRMRankingTool

log = logging.getLogger(__name__)


class PairRMGroupRankingFeature(BaseGroupFeature):
    """
    Group-wise PairRM ranking feature.

    - Groups rows by a configurable key (e.g. 'group_id' or 'seed_id')
    - For each group, uses PairRMRankingTool to score candidate texts
      given a shared context
    - Attaches per-row:
        - pairrm_score  (numeric reward)
        - pairrm_rank   (0 = best)
        - pairrm_winner (bool: True for best in group)
        - optional: pairrm_group (the group id)
    - Optionally reorders rows within each group by rank.

    Typical row layout in the metrics pipeline:

        row = {
            "group_id": <some stable id>,        # used for grouping
            "blog_text": "...",                  # candidate text
            "context_text": "...",               # optional shared context
            ...
        }
    """

    name = "pairrm_group_ranking"
    # 'requires' can be used if you want to enforce upstream dependencies
    requires: List[str] = []

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Try to mirror how FrontierLensGroupFeature wires its tool config.
        # You can either:
        #   - pass a dedicated cfg["tools"]["pairrm_ranker"] section, or
        #   - pass the whole cfg and let the tool pick what it needs.
        tool_cfg = cfg.get("tools", {}).get("pairrm_ranker", cfg)
        self.tool = PairRMRankingTool(tool_cfg, memory, container, logger)

        self.enabled: bool = bool(cfg.get("enabled", True))

        # Group configuration
        self.group_key: str = cfg.get("group_key", "group_id")

        # Where to find the candidate text on each row
        self.candidate_text_column: str = cfg.get(
            "candidate_text_column", "blog_text"
        )

        # Where to find context text (optional).
        # If None, we fall back to context["pairrm_context"] or "".
        self.context_text_column: Optional[str] = cfg.get(
            "context_text_column", "context_text"
        )

        # Column used as candidate id (for traceability)
        self.candidate_id_column: Optional[str] = cfg.get(
            "candidate_id_column", "id"
        )

        # Whether to reorder rows within each group by rank
        self.order_within_groups: bool = bool(
            cfg.get("order_within_groups", False)
        )

        # Output keys
        self.score_key: str = cfg.get("score_key", "pairrm_score")
        self.rank_key: str = cfg.get("rank_key", "pairrm_rank")
        self.winner_flag_key: str = cfg.get("winner_flag_key", "pairrm_winner")
        self.group_label_key: str = cfg.get("group_label_key", "pairrm_group")

        # Basic telemetry
        self._groups_seen: int = 0
        self._groups_ranked: int = 0
        self._rows_in: int = 0
        self._rows_scored: int = 0
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseGroupFeature API
    # ------------------------------------------------------------------

    async def apply(
        self, rows: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        rows: list of row dicts (same as FrontierLensGroupFeature)
        context: pipeline-level context (can include pipeline_run_id, etc.)

        Returns:
            rows, optionally reordered within groups.
        """
        if not self.enabled:
            return rows

        self._reset_telemetry(len(rows))

        if not rows:
            return rows

        # Group rows by group_key
        grouped: DefaultDict[Any, List[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            group_val = row.get(self.group_key)
            if group_val is None:
                # Treat rows without a group_key as their own singleton groups
                group_val = f"__row_{idx}"
            grouped[group_val].append(idx)

        # If we are going to reorder within groups, we will build a new list
        new_rows: List[Dict[str, Any]] = [] if self.order_within_groups else rows

        # Optional context override from pipeline-level context
        pipeline_context_text: str = context.get("pairrm_context", "") or ""

        for group_id, indices in grouped.items():
            self._groups_seen += 1

            group_rows = [rows[i] for i in indices]

            # Determine context text for this group
            context_text = self._resolve_context_text(
                group_rows, pipeline_context_text
            )

            # Build candidate list
            candidate_texts: List[str] = []
            candidate_ids: List[str] = []
            idx_map: List[int] = []  # mapping from candidate index -> row index

            for row_index, row in zip(indices, group_rows):
                text = row.get(self.candidate_text_column)
                if not isinstance(text, str) or not text.strip():
                    continue

                cid = None
                if self.candidate_id_column is not None:
                    cid = row.get(self.candidate_id_column)
                if cid is None:
                    cid = f"{group_id}:{row_index}"

                candidate_texts.append(text)
                candidate_ids.append(str(cid))
                idx_map.append(row_index)

            if len(candidate_texts) < 2:
                # Not enough candidates to rank – just attach defaults and move on
                for row_index in indices:
                    row = rows[row_index]
                    row.setdefault(self.score_key, None)
                    row.setdefault(self.rank_key, None)
                    row.setdefault(self.winner_flag_key, False)
                    row.setdefault(self.group_label_key, str(group_id))
                if self.order_within_groups:
                    new_rows.extend(group_rows)
                continue

            # We have at least 2 candidates – call the PairRM tool
            try:
                result = self.tool.rank_candidates(
                    context_text=context_text,
                    candidates=candidate_texts,
                    candidate_ids=candidate_ids,
                )
            except Exception as e:
                # Fail-closed: leave rows unchanged for this group
                self._error = f"{type(e).__name__}: {e}"
                log.warning(
                    "[PairRMGroupRankingFeature] Failed ranking group=%r: %s",
                    group_id,
                    self._error,
                )
                for row_index in indices:
                    row = rows[row_index]
                    row.setdefault(self.score_key, None)
                    row.setdefault(self.rank_key, None)
                    row.setdefault(self.winner_flag_key, False)
                    row.setdefault(self.group_label_key, str(group_id))
                if self.order_within_groups:
                    new_rows.extend(group_rows)
                continue

            # Attach scores/ranks
            self._groups_ranked += 1
            scored_candidates = result.get("candidates", [])
            winner_id: Optional[str] = result.get("winner_id")

            scores_by_id: Dict[str, float] = {
                str(c["id"]): float(c["score"])
                for c in scored_candidates
            }
            ranks_by_id: Dict[str, int] = {
                str(c["id"]): rank
                for rank, c in enumerate(scored_candidates)
            }

            for local_idx, row_index in enumerate(idx_map):
                row = rows[row_index]
                cid = str(candidate_ids[local_idx])

                score = scores_by_id.get(cid)
                rank = ranks_by_id.get(cid)

                row[self.score_key] = score
                row[self.rank_key] = rank
                row[self.winner_flag_key] = (winner_id is not None and cid == winner_id)
                row.setdefault(self.group_label_key, str(group_id))

                if score is not None:
                    self._rows_scored += 1

            # Optionally reorder rows within this group by rank
            if self.order_within_groups:
                indices_sorted = sorted(
                    idx_map,
                    key=lambda ri: rows[ri].get(self.rank_key, 1e9),
                )
                for ri in indices_sorted:
                    new_rows.append(rows[ri])

        return new_rows if self.order_within_groups else rows

    def _resolve_context_text(
        self,
        group_rows: List[Dict[str, Any]],
        pipeline_context_text: str,
    ) -> str:
        """
        Decide which context text to use for a group.

        Priority:
          1. First non-empty row[self.context_text_column] if configured.
          2. pipeline_context["pairrm_context"] if provided.
          3. Empty string.
        """
        if self.context_text_column:
            for r in group_rows:
                ctx = r.get(self.context_text_column)
                if isinstance(ctx, str) and ctx.strip():
                    return ctx
        return pipeline_context_text

    def _reset_telemetry(self, rows_in: int) -> None:
        self._groups_seen = 0
        self._groups_ranked = 0
        self._rows_in = rows_in
        self._rows_scored = 0
        self._error = None

    def report(self) -> Dict[str, Any]:
        """
        Optional report method, mirroring FrontierLensGroupFeature.
        """
        ok = self._error is None
        return {
            "feature": self.name,
            "ok": ok,
            "rows_in": self._rows_in,
            "rows_scored": self._rows_scored,
            "groups_seen": self._groups_seen,
            "groups_ranked": self._groups_ranked,
            "error": self._error,
            "summary": (
                f"rows_scored={self._rows_scored}/{self._rows_in}; "
                f"groups_ranked={self._groups_ranked}/{self._groups_seen}"
            ),
        }
Laura