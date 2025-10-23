from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from stephanie.components.gap.models import TripleSample

_logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """
    Source selection + knobs for fetching training/eval triples.

    source: "memory" | "jsonl" | "parquet" | "custom"
    limit: hard cap per dimension (post-filter)
    max_text_len: if set, skip items where goal or output exceeds this length
    overfetch_factor: for 'memory' source, multiply the fetch size to
                      improve chances of meeting 'limit' after filtering
    """
    source: str = "memory"
    limit: int = 100
    max_text_len: Optional[int] = 2000
    overfetch_factor: int = 5

    # file paths only used for jsonl/parquet sources
    jsonl_path: Optional[str] = None
    parquet_path: Optional[str] = None


class DataRetriever:
    """
    Pluggable abstraction for fetching (goal_text, output_text, target) triples by dimension.
    Default implementation pulls from PreferencePairBuilder (memory).
    """

    def __init__(self, container, logger, retriever_cfg: RetrieverConfig | None = None):
        self.container = container
        self.logger = logger or _logger
        self.cfg = retriever_cfg or RetrieverConfig()

    async def get_triples_by_dimension(
        self,
        dimensions: List[str],
        *,
        memory=None,
        limit: Optional[int] = None,
    ) -> Dict[str, List[TripleSample]]:
        """
        Returns a dict: { dimension -> List[TripleSample] }.
        Dispatches based on configured source. Applies max_text_len filtering,
        and keeps collecting until post-filter count reaches 'limit'.
        """
        src = (self.cfg.source or "memory").lower()
        cap = int(limit if limit is not None else self.cfg.limit)

        if src == "memory":
            return await self._from_memory(dimensions, memory=memory, limit=cap)

        if src == "jsonl":
            if not self.cfg.jsonl_path:
                raise ValueError("DataRetriever: jsonl_path is required for source='jsonl'")
            return self._from_jsonl(dimensions, jsonl_path=self.cfg.jsonl_path, limit=cap)

        if src == "parquet":
            if not self.cfg.parquet_path:
                raise ValueError("DataRetriever: parquet_path is required for source='parquet'")
            return self._from_parquet(dimensions, parquet_path=self.cfg.parquet_path, limit=cap)

        if src == "custom":
            fn = self.container.get("gap_custom_retriever", None)
            if not fn:
                raise ValueError("DataRetriever: no 'gap_custom_retriever' registered on container")
            return await fn(dimensions=dimensions, limit=cap, logger=self.logger)

        raise ValueError(f"DataRetriever: unknown source '{src}'")

    # --- Implementations -----------------------------------------------------

    async def _from_memory(
        self,
        dimensions: List[str],
        *,
        memory,
        limit: int,
    ) -> Dict[str, List[TripleSample]]:
        """
        Pull pairs from PreferencePairBuilder and flatten to TripleSample.
        We over-fetch to preserve 'limit' after filtering long texts.
        """
        from stephanie.scoring.training.preference_pair_builder import \
            PreferencePairBuilder
        ppb = PreferencePairBuilder(memory, self.logger)

        out: Dict[str, List[TripleSample]] = {}
        over = max(1, int(self.cfg.overfetch_factor))
        fetch_n = max(limit * over, limit)

        for dim in dimensions:
            # Overfetch; we'll filter and then slice to 'limit'
            pairs = ppb.get_training_pairs_by_dimension(dimension=dim, limit=fetch_n)
            samples = pairs.get(dim, [])
            triples = self._collect_with_filters(samples, dim, limit)
            out[dim] = triples
            self.logger.log("DataRetrieverMemory", {"dimension": dim, "requested": fetch_n, "returned": len(triples)})
            if len(triples) < limit:
                _logger.warning(
                    "DataRetrieverMemoryShortfall: dim=%s requested=%d got=%d (after filtering)",
                    dim, limit, len(triples)
                )
        return out

    def _from_jsonl(
        self,
        dimensions: List[str],
        *,
        jsonl_path: str,
        limit: int,
    ) -> Dict[str, List[TripleSample]]:
        import json
        want = {d: limit for d in dimensions}
        got: Dict[str, List[TripleSample]] = {d: [] for d in dimensions}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            idx_by_dim = {d: 0 for d in dimensions}
            for line in f:
                if all(len(got[d]) >= want[d] for d in dimensions):
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                dim = str(row.get("dimension", "")).strip()
                if dim not in got or len(got[dim]) >= want[dim]:
                    continue
                ts = self._row_to_triple(row, dim, idx=idx_by_dim[dim])
                idx_by_dim[dim] += 1
                if ts and self._within_len(ts.goal_text, ts.output_text):
                    got[dim].append(ts)

        self.logger.log("DataRetrieverJSONL", {d: len(v) for d, v in got.items()})
        return got

    def _from_parquet(
        self,
        dimensions: List[str],
        *,
        parquet_path: str,
        limit: int,
    ) -> Dict[str, List[TripleSample]]:
        import pandas as pd
        df = pd.read_parquet(parquet_path)

        want = {d: limit for d in dimensions}
        got: Dict[str, List[TripleSample]] = {d: [] for d in dimensions}

        for d in dimensions:
            sub = df[df["dimension"] == d]
            idx = 0
            for _, row in sub.iterrows():
                if len(got[d]) >= want[d]:
                    break
                ts = self._row_to_triple(row.to_dict(), d, idx=idx)
                idx += 1
                if ts and self._within_len(ts.goal_text, ts.output_text):
                    got[d].append(ts)

        self.logger.log("DataRetrieverParquet", {d: len(v) for d, v in got.items()})
        return got

    # --- Flatten & collect helpers ------------------------------------------

    def _collect_with_filters(self, samples: List[Dict[str, Any]], dimension: str, limit: int) -> List[TripleSample]:
        """
        Flattens a mixed list of sample shapes into TripleSample, applying max_text_len
        filtering and stopping when 'limit' is reached.
        """
        triples: List[TripleSample] = []
        for i, s in enumerate(samples):
            goal = (s.get("goal_text") or s.get("title") or "").strip()

            # singleton
            if "output" in s and ("score" in s or "target_score" in s):
                out = (s.get("output") or "").strip()
                val = s.get("target_score", s.get("score"))
                if goal and out and val is not None and self._within_len(goal, out):
                    triples.append(
                        TripleSample(
                            node_id=f"{dimension}|{i:06d}",
                            dimension=dimension,
                            goal_text=goal,
                            output_text=out,
                            target_value=float(val),
                            fingerprint=self._fingerprint(goal, out),
                        )
                    )

            # pairwise
            elif all(k in s for k in ("output_a", "output_b", "value_a", "value_b")):
                for suf in ("a", "b"):
                    out = (s.get(f"output_{suf}") or "").strip()
                    val = s.get(f"value_{suf}")
                    if goal and out and val is not None and self._within_len(goal, out):
                        triples.append(
                            TripleSample(
                                node_id=f"{dimension}|{i:06d}_{suf}",
                                dimension=dimension,
                                goal_text=goal,
                                output_text=out,
                                target_value=float(val),
                                fingerprint=self._fingerprint(goal, out),
                            )
                        )

            # explicit MRQ-ish form
            elif ("goal_text" in s and "scorable_text" in s and ("target_score" in s or "score" in s)):
                out = (s.get("scorable_text") or "").strip()
                val = s.get("target_score", s.get("score"))
                if goal and out and val is not None and self._within_len(goal, out):
                    triples.append(
                        TripleSample(
                            node_id=f"{dimension}|{i:06d}",
                            dimension=dimension,
                            goal_text=goal,
                            output_text=out,
                            target_value=float(val),
                            fingerprint=self._fingerprint(goal, out),
                        )
                    )

            if len(triples) >= limit:
                break
        return triples

    def _row_to_triple(self, row: Dict[str, Any], dimension: str, idx: int) -> TripleSample | None:
        goal = (row.get("goal_text") or row.get("title") or "").strip()
        out = (row.get("output_text") or row.get("output") or "").strip()
        val = row.get("target") or row.get("target_score") or row.get("score")
        if not goal or not out:
            return None
        return TripleSample(
            node_id=f"{dimension}|{idx:06d}",
            dimension=dimension,
            goal_text=goal,
            output_text=out,
            target_value=float(val) if val is not None else 0.0,
            fingerprint=self._fingerprint(goal, out),
        )

    # --- Utils ---------------------------------------------------------------

    def _within_len(self, goal_text: str, output_text: str) -> bool:
        m = self.cfg.max_text_len
        if m is None:
            return True
        try:
            return (len(goal_text) <= m) and (len(output_text) <= m)
        except Exception:
            return False

    def _fingerprint(self, goal_text: str, output_text: str) -> str:
        import hashlib
        payload = (goal_text.strip() + "\n␟\n" + output_text.strip()).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()
