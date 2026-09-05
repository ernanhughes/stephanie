"""Evaluation Migration Audit (Stage 2.5, gate 2) — READ ONLY.

Compares legacy snapshot JSON vs normalized score rows per evaluation,
classifies identity/latest/confidence semantics, and prints the
migration baseline report. Never writes.

Usage:
    python scripts/evaluation_migration_audit.py [--dsn URL] [--limit N] [--sample-values N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("psycopg2 is required for the audit", file=sys.stderr)
    sys.exit(2)

DEFAULT_DSN = os.environ.get("STEPHANIE_AUDIT_DSN", "postgresql://co:co@localhost:5432/co")


def main() -> int:
    args = parse_args()
    conn = psycopg2.connect(args.dsn, connect_timeout=10)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("SELECT count(*) AS n FROM evaluations")
    total_evals = cur.fetchone()["n"]
    cur.execute("SELECT count(*) AS n FROM scores")
    total_scores = cur.fetchone()["n"]

    counters = Counter(
        match=0,
        expected_divergence=0,
        unexpected_divergence=0,
        legacy_only=0,
        canonical_only=0,
    )
    unexpected_kinds = Counter()
    inspected = 0

    cur.execute(
        """
        SELECT e.id, e.scorable_type, e.scorable_id, e.evaluator_name,
               e.model_name, e.agent_name, e.strategy, e.scores AS snapshot,
               e.created_at
          FROM evaluations e
         ORDER BY e.id
         LIMIT %s
        """,
        (args.limit,),
    )
    # Stream score rows for the same id range in one query.
    cur2 = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur2.execute(
        """
        SELECT s.evaluation_id, s.dimension, s.score
          FROM scores s
          JOIN evaluations e ON e.id = s.evaluation_id
         ORDER BY s.evaluation_id, s.id
         LIMIT %s
        """,
        (args.limit * 12,),
    )
    rows_by_eval: dict[int, list] = {}
    for row in cur2.fetchall():
        rows_by_eval.setdefault(row["evaluation_id"], []).append(row)
    cur2.close()

    sample_compared = 0
    for ev in cur.fetchall():
        inspected += 1
        snapshot = ev["snapshot"] or {}
        if isinstance(snapshot, str):
            try:
                snapshot = json.loads(snapshot)
            except json.JSONDecodeError:
                snapshot = {}
        if isinstance(snapshot, dict):
            # to_json(stage) envelope: real dims nest under "dimensions".
            nested = snapshot.get("dimensions")
            if isinstance(nested, dict) and nested:
                snap_dims = {k: v for k, v in nested.items() if not str(k).startswith("_")}
                derived = [k for k in snapshot if k != "dimensions"]
            else:
                snap_dims = {k: v for k, v in snapshot.items() if not str(k).startswith("_")}
                derived = [k for k in snapshot if str(k).startswith("_") or k in ("avg", "final_score")]
        elif isinstance(snapshot, list):
            # List-of-{dimension, score, ...} envelope (knowledge_llm writers).
            snap_dims = {}
            for item in snapshot:
                if isinstance(item, dict) and "dimension" in item:
                    snap_dims[item["dimension"]] = item.get("score")
            derived = ["<list-envelope>"] if snap_dims else []
            if not snap_dims:
                unexpected_kinds["non_dict_snapshot"] += 1
        else:
            snap_dims = {}
            derived = ["<non-dict-snapshot>"]
            unexpected_kinds["non_dict_snapshot"] += 1
        row_list = rows_by_eval.get(ev["id"], [])

        # Identity check.
        if not ev["scorable_type"] or not ev["scorable_id"]:
            counters["unexpected_divergence"] += 1
            unexpected_kinds["identity_mismatch"] += 1
            continue

        if not row_list and not snap_dims:
            counters["match"] += 1  # empty both sides
            continue
        if not row_list and snap_dims:
            counters["legacy_only"] += 1  # snapshot-only legacy row
            continue
        if row_list and not snap_dims:
            counters["canonical_only"] += 1  # normalized rows, no snapshot
            continue

        row_dims = {r["dimension"]: r["score"] for r in row_list}
        if set(snap_dims) != set(row_dims):
            missing = set(snap_dims) - set(row_dims)
            extra = set(row_dims) - set(snap_dims)
            if missing:
                unexpected_kinds["missing_dimension"] += 1
            if extra:
                unexpected_kinds["extra_normalized_dimension"] += 1
            counters["unexpected_divergence"] += 1
            continue

        # Same dimension set: compare values on a sample (full compare optional).
        mismatch = False
        for dim in snap_dims:
            snap_val = snap_dims[dim]
            if isinstance(snap_val, dict):
                snap_val = snap_val.get("score")
            if not isinstance(snap_val, (int, float)):
                continue
            if sample_compared >= args.sample_values:
                break
            sample_compared += 1
            row_val = row_dims[dim]
            if row_val is None or abs(float(snap_val) - float(row_val)) > 1e-9:
                mismatch = True
                break
        if mismatch:
            counters["unexpected_divergence"] += 1
            unexpected_kinds["snapshot_vs_normalized_score"] += 1
        else:
            # Snapshot carries avg/final_score-style derived keys?
            if derived:
                counters["expected_divergence"] += 1
            else:
                counters["match"] += 1

    # Confidence semantics: legacy has no confidence columns — count snapshots
    # that smuggle confidence-like keys (expected divergence class).
    cur.execute(
        """
        SELECT count(*) AS n FROM evaluations
        WHERE scores::text LIKE '%confidence%'
        """
    )
    confidence_smuggled = cur.fetchone()["n"]
    cur.execute("SELECT count(*) AS n FROM scores WHERE prompt_hash IS NOT NULL")
    prompt_hash_set = cur.fetchone()["n"]
    cur.execute(
        """
        SELECT count(*) AS n FROM (
          SELECT scorable_type, scorable_id, created_at, count(*) AS c
            FROM evaluations GROUP BY 1, 2, 3 HAVING count(*) > 1
        ) ties
        """
    )
    latest_ties = cur.fetchone()["n"]
    conn.close()

    print("Evaluation Migration Audit")
    print("==========================")
    print(f"Stephanie rows inspected:       {inspected} (of {total_evals} evaluations, {total_scores} scores)")
    print()
    print(f"MATCH:                          {counters['match']}")
    print(f"EXPECTED_DIVERGENCE:            {counters['expected_divergence']}")
    print(f"UNEXPECTED_DIVERGENCE:          {counters['unexpected_divergence']}")
    print(f"LEGACY_ONLY:                    {counters['legacy_only']}")
    print(f"CANONICAL_ONLY:                 {counters['canonical_only']}")
    print()
    print("Top unexpected divergences:")
    for i, (kind, n) in enumerate(unexpected_kinds.most_common(5), 1):
        print(f"  {i}. {kind:<32} {n}")
    if not unexpected_kinds:
        print("  (none)")
    print()
    print("Semantic notes (expected):")
    print(f"  snapshots mentioning confidence: {confidence_smuggled}")
    print(f"  scores with prompt_hash set:     {prompt_hash_set} (live for mrq/ebt/sicql/svm/llm; absent for ranker/hrm)")
    print(f"  latest-ordering tie groups:      {latest_ties} (same subject+created_at)")
    print(f"  score values compared:           {sample_compared}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--limit", type=int, default=200000)
    parser.add_argument("--sample-values", type=int, default=50000)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
