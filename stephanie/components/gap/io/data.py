# stephanie/components/gap/io/data.py
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from stephanie.scoring.training.preference_pair_builder import PreferencePairBuilder
from stephanie.utils.json_sanitize import dumps_safe

def _fingerprint(goal_text: str, output_text: str) -> str:
    h = hashlib.sha1()
    h.update((goal_text.strip() + "\n␟\n" + output_text.strip()).encode("utf-8"))
    return h.hexdigest()

def _flatten_samples_for_eval(samples: List[dict]) -> List[Tuple[str, str, float]]:
    triples = []
    for s in samples:
        title = (s.get("goal_text") or s.get("title") or "").strip()
        # singleton
        if "output" in s and ("score" in s or "target_score" in s):
            out = (s.get("output") or "").strip()
            val = s.get("target_score", s.get("score"))
            if title and out and val is not None:
                triples.append((title, out, float(val)))
            continue
        # pairwise
        if all(k in s for k in ("output_a","output_b","value_a","value_b")):
            a_out, b_out = (s.get("output_a") or "").strip(), (s.get("output_b") or "").strip()
            a_val, b_val = s.get("value_a", None), s.get("value_b", None)
            if title and a_out and a_val is not None:
                triples.append((title, a_out, float(a_val)))
            if title and b_out and b_val is not None:
                triples.append((title, b_out, float(b_val)))
            continue
        # explicit
        if ("goal_text" in s and "scorable_text" in s and ("target_score" in s or "score" in s)):
            out = (s.get("scorable_text") or "").strip()
            val = s.get("target_score", s.get("score"))
            if title and out and val is not None:
                triples.append((title, out, float(val)))
    return triples

def _dedupe_triples_by_dimension(
    triples_by_dim: Dict[str, List[Tuple[str,str,float]]],
    policy: str = "first_wins",
    per_dim_cap: int | None = None
) -> Dict[str, List[Tuple[str,str,float]]]:
    dims = list(triples_by_dim.keys())
    if policy == "first_wins":
        seen: set[str] = set()
        out = {d: [] for d in dims}
        for d in dims:
            for (g,o,v) in triples_by_dim[d]:
                key = _fingerprint(g,o)
                if key in seen:
                    continue
                out[d].append((g,o,v))
                seen.add(key)
            if per_dim_cap is not None and len(out[d]) > per_dim_cap:
                out[d] = out[d][:per_dim_cap]
        return out
    elif policy == "round_robin":
        pool: Dict[str, Tuple[str,str,float]] = {}
        for d in dims:
            for (g,o,v) in triples_by_dim[d]:
                key = _fingerprint(g,o)
                if key not in pool:
                    pool[key] = (g,o,v)
        keys = list(pool.keys())
        out = {d: [] for d in dims}
        i = 0
        for k in keys:
            d = dims[i % len(dims)]
            if per_dim_cap is None or len(out[d]) < per_dim_cap:
                out[d].append(pool[k]); i += 1
        return out
    else:
        raise ValueError(f"Unknown policy: {policy}")

@dataclass
class TriplesIndex:
    jsonl_path: str
    parquet_path: str | None
    head_path: str
    rows: int
    per_dim_counts: Dict[str, int]

class DataRetriever:
    """
    Pull triples from memory (PreferencePairBuilder), dedupe/cap, and
    materialize raw/triples.jsonl (+head/parquet). Processors downstream
    only need the path to jsonl/parquet.
    """
    def __init__(self, base_dir: str, logger):
        self.base_dir = base_dir
        self.logger = logger

    def build_triples_index(
        self,
        *,
        run_id: str,
        dimensions: List[str],
        memory,
        policy: str = "first_wins",
        per_dim_cap: int | None = None,
        per_dim_limit: int | None = None
    ) -> TriplesIndex:
        raw_dir = os.path.join(self.base_dir, run_id, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        pair_builder = PreferencePairBuilder(memory, self.logger)
        triples_by_dim: Dict[str, List[Tuple[str,str,float]]] = {}
        total_raw = 0

        for d in dimensions:
            # ⚠️ limit guards retrieval time; set to None for full pull
            pairs_by_dim = pair_builder.get_training_pairs_by_dimension(
                dimension=d,
                limit=per_dim_limit
            )
            samples_full = pairs_by_dim.get(d, [])
            if not samples_full:
                self.logger.log("GapNoSamplesFound", {"dimension": d})
                triples_by_dim[d] = []
                continue
            triples = _flatten_samples_for_eval(samples_full)
            triples_by_dim[d] = triples
            total_raw += len(triples)

        deduped = _dedupe_triples_by_dimension(
            triples_by_dim, policy=policy, per_dim_cap=per_dim_cap
        )

        # Write raw/triples.jsonl (+head + parquet)
        jsonl_path = os.path.join(raw_dir, "triples.jsonl")
        head_path  = os.path.join(raw_dir, "triples.head.json")
        parquet_path = os.path.join(raw_dir, "triples.parquet")

        rows: List[Dict[str, Any]] = []
        per_dim_counts: Dict[str,int] = {}
        for d, triples in deduped.items():
            per_dim_counts[d] = len(triples)
            for i, (g, o, v) in enumerate(triples):
                rows.append({
                    "node_id": f"{run_id}|{d}|{i:06d}",
                    "dimension": d,
                    "goal_text": g,
                    "output_text": o,
                    "target": float(v) if v is not None else None,
                    "turn_idx": i,
                    "fingerprint": _fingerprint(g,o),
                    "policy": policy,
                })

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(dumps_safe(r)); f.write("\n")
        with open(head_path, "w", encoding="utf-8") as f:
            f.write(dumps_safe(rows[:200], indent=2))

        try:
            pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = None

        self.logger.log("GapTriplesIndex", {
            "run_id": run_id,
            "path": jsonl_path,
            "rows": len(rows),
            "per_dim_counts": per_dim_counts
        })

        return TriplesIndex(
            jsonl_path=jsonl_path,
            parquet_path=parquet_path,
            head_path=head_path,
            rows=len(rows),
            per_dim_counts=per_dim_counts,
        )
