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
    limit: optional hard cap per dimension
    """
    source: str = "memory"
<<<<<<< HEAD
=======
    # limit: int = 1000 # CAP count limit per dimension
>>>>>>> main
    limit: int = 100
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
        Dispatches based on configured source.
        """
        src = (self.cfg.source or "memory").lower()
        cap = limit if limit is not None else self.cfg.limit

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

        # Hook for custom app-specific retrieval (e.g., DB)
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
        limit: Optional[int],
    ) -> Dict[str, List[TripleSample]]:
        """
        Pull pairs from PreferencePairBuilder and flatten to TripleSample.
        """
        from stephanie.scoring.training.preference_pair_builder import PreferencePairBuilder
        ppb = PreferencePairBuilder(memory, self.logger)

        out: Dict[str, List[TripleSample]] = {}
        for dim in dimensions:
            pairs = ppb.get_training_pairs_by_dimension(dimension=dim, limit=limit)
            samples = pairs.get(dim, [])
            triples = self._flatten_samples(samples, dim)
            if limit is not None:
                triples = triples[:limit]
            out[dim] = triples
            self.logger.log("DataRetrieverMemory", {"dimension": dim, "count": len(triples)})
        return out

    def _from_jsonl(
        self,
        dimensions: List[str],
        *,
        jsonl_path: str,
        limit: Optional[int],
    ) -> Dict[str, List[TripleSample]]:
        import json
        out: Dict[str, List[TripleSample]] = {d: [] for d in dimensions}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                dim = str(row.get("dimension", "")).strip()
                if dim not in out:
                    continue
                ts = self._row_to_triple(row, dim, idx=len(out[dim]))
                if ts:
                    out[dim].append(ts)
        if limit is not None:
            for d in out:
                out[d] = out[d][:limit]
        self.logger.log("DataRetrieverJSONL", {d: len(v) for d, v in out.items()})
        return out

    def _from_parquet(
        self,
        dimensions: List[str],
        *,
        parquet_path: str,
        limit: Optional[int],
    ) -> Dict[str, List[TripleSample]]:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        out: Dict[str, List[TripleSample]] = {d: [] for d in dimensions}
        # expected columns: dimension, goal_text, output_text, (optional) target
        for d in dimensions:
            sub = df[df["dimension"] == d]
            for i, row in enumerate(sub.itertuples(index=False)):
                ts = self._row_to_triple(row._asdict() if hasattr(row, "_asdict") else row.__dict__, d, idx=i)
                if ts:
                    out[d].append(ts)
        if limit is not None:
            for d in out:
                out[d] = out[d][:limit]
        self.logger.log("DataRetrieverParquet", {d: len(v) for d, v in out.items()})
        return out

    # --- Flatten helpers -----------------------------------------------------

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

    def _flatten_samples(self, samples: List[Dict[str, Any]], dimension: str) -> List[TripleSample]:
        triples: List[TripleSample] = []
        for i, s in enumerate(samples):
            goal = (s.get("goal_text") or s.get("title") or "").strip()

            # singleton
            if "output" in s and ("score" in s or "target_score" in s):
                out = (s.get("output") or "").strip()
                val = s.get("target_score", s.get("score"))
                if goal and out and val is not None:
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
                continue

            # pairwise
            if all(k in s for k in ("output_a", "output_b", "value_a", "value_b")):
                for suf in ("a", "b"):
                    out = (s.get(f"output_{suf}") or "").strip()
                    val = s.get(f"value_{suf}")
                    if goal and out and val is not None:
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
                continue

            # explicit MRQ-ish form
            if ("goal_text" in s and "scorable_text" in s and ("target_score" in s or "score" in s)):
                out = (s.get("scorable_text") or "").strip()
                val = s.get("target_score", s.get("score"))
                if goal and out and val is not None:
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
        return triples

    def _fingerprint(self, goal_text: str, output_text: str) -> str:
        import hashlib
        payload = (goal_text.strip() + "\n␟\n" + output_text.strip()).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()
