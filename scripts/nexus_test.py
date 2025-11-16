#!/usr/bin/env python
"""
Nexus graph performance benchmark.

- Creates Nexus-style tables (nexus_scorable, nexus_metrics, nexus_edge)
- Bulk-inserts synthetic nodes, metrics, and edges using COPY
- Runs a "Pulse" benchmark: random src node → neighbors → metrics aggregate

Run on a scratch database (e.g. test_nexus_db) – this script DROPs tables.
"""

import json
import random
import time
import uuid
from io import StringIO
from typing import List, Tuple

import psycopg2
from psycopg2.extensions import connection as PGConnection

# --------------------------------------------------------------------------- #
# CONFIG – tune these for your box                                            #
# --------------------------------------------------------------------------- #

DB_NAME = "co"
DB_USER = "co"
DB_PASSWORD = "co"  # <<< CHANGE THIS
DB_HOST = "localhost"
DB_PORT = 5432

# Scale knobs (start smaller: e.g. 50_000 / 100_000 / 1_000)
RUN_ID = "bench_run_1"
NUM_NODES = 500_000      # synthetic "thoughts"
NUM_EDGES = 1_000_000    # synthetic relationships
NUM_PULSES = 5_000       # random traversals to measure QPS

# --------------------------------------------------------------------------- #
# Connection + setup                                                          #
# --------------------------------------------------------------------------- #

def get_db_connection() -> PGConnection:
    """
    Connect to Postgres using basic credentials.
    Uses explicit transactions (autocommit = False) for bulk loading.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    conn.autocommit = False
    return conn


def setup_nexus_tables(conn: PGConnection) -> None:
    """
    Drop and recreate minimal Nexus-style tables:
      - nexus_scorable
      - nexus_metrics
      - nexus_edge

    These mirror the shape of your ORM enough to get realistic row sizes and
    traversal patterns.
    """
    cur = conn.cursor()
    print("Setting up Nexus tables (DROP + CREATE)...")

    cur.execute("DROP TABLE IF EXISTS nexus_edge CASCADE;")
    cur.execute("DROP TABLE IF EXISTS nexus_metrics CASCADE;")
    cur.execute("DROP TABLE IF EXISTS nexus_scorable CASCADE;")

    # NexusScorableORM-like table
    cur.execute(
        """
        CREATE TABLE nexus_scorable (
            id          VARCHAR PRIMARY KEY,
            created_ts  TIMESTAMPTZ DEFAULT NOW(),
            chat_id     VARCHAR NULL,
            turn_index  INTEGER NULL,
            target_type VARCHAR NULL,
            text        TEXT NULL,
            domains     JSONB NULL,
            entities    JSONB NULL,
            meta        JSONB NULL
        );
        """
    )

    # NexusMetricsORM-like table
    # We store a small vector with a "coverage" dimension for the Pulse query.
    cur.execute(
        """
        CREATE TABLE nexus_metrics (
            scorable_id VARCHAR PRIMARY KEY,
            columns     JSONB NOT NULL,
            values      JSONB NOT NULL,
            vector      JSONB NULL
        );
        """
    )

    # NexusEdgeORM-like table
    cur.execute(
        """
        CREATE TABLE nexus_edge (
            run_id    VARCHAR NOT NULL,
            src       VARCHAR NOT NULL,
            dst       VARCHAR NOT NULL,
            type      VARCHAR NOT NULL,
            weight    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            channels  JSONB NULL,
            created_ts TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (run_id, src, dst, type)
        );
        """
    )

    # Traversal-critical indexes
    cur.execute("CREATE INDEX idx_nexus_edge_src ON nexus_edge (run_id, src);")
    cur.execute("CREATE INDEX idx_nexus_edge_dst ON nexus_edge (run_id, dst);")

    conn.commit()
    cur.close()
    print("Tables created.\n")


# --------------------------------------------------------------------------- #
# Data generation + bulk load                                                 #
# --------------------------------------------------------------------------- #

NODE_IDS: List[str] = []


def run_bulk_graph_construction(conn: PGConnection) -> Tuple[float, float]:
    """
    Generate NUM_NODES + NUM_EDGES synthetic rows and bulk insert them
    using COPY. Returns (duration_seconds, rows_per_second).
    """
    global NODE_IDS

    cur = conn.cursor()
    print(
        f"Starting bulk write test: "
        f"{NUM_NODES:,} nodes, {NUM_NODES:,} metrics, {NUM_EDGES:,} edges..."
    )

    # --- 1. Generate node + metrics data in-memory -------------------------

    NODE_IDS = [str(uuid.uuid4()) for _ in range(NUM_NODES)]

    node_lines: List[str] = []
    metrics_lines: List[str] = []
    edge_lines: List[str] = []

    domains_pool = ["code", "math", "nlp", "vision", "planning"]
    types_pool = ["document", "turn", "summary"]
    edge_types = ["knn_global", "temporal_next", "shared_domain"]

    for i, node_id in enumerate(NODE_IDS):
        chat_id = f"chat-{random.randint(1, 10_000)}"
        turn_index = random.randint(0, 2048)
        target_type = random.choice(types_pool)
        text = f"synthetic thought fragment #{i}"
        domains = [random.choice(domains_pool)]
        entities = [f"entity-{random.randint(1, 5000)}"]
        meta = {"run_id": RUN_ID, "phase": "bench"}

        node_lines.append(
            "\t".join(
                [
                    node_id,
                    chat_id,
                    str(turn_index),
                    target_type,
                    text,
                    json.dumps(domains, ensure_ascii=False),
                    json.dumps(entities, ensure_ascii=False),
                    json.dumps(meta, ensure_ascii=False),
                ]
            )
            + "\n"
        )

        # Simple metrics: we care about "coverage" for the pulse
        coverage = random.uniform(0.1, 0.9)
        risk = random.uniform(0.0, 1.0)
        cols = ["coverage", "risk"]
        vals = [coverage, risk]
        vec = {"coverage": coverage, "risk": risk}

        metrics_lines.append(
            "\t".join(
                [
                    node_id,
                    json.dumps(cols, ensure_ascii=False),
                    json.dumps(vals, ensure_ascii=False),
                    json.dumps(vec, ensure_ascii=False),
                ]
            )
            + "\n"
        )

    # Edges: random links within the node set
    for _ in range(NUM_EDGES):
        src = random.choice(NODE_IDS)
        dst = random.choice(NODE_IDS)
        etype = random.choice(edge_types)
        weight = random.uniform(0.0, 1.0)
        channels = {"source": "bench", "etype": etype}

        edge_lines.append(
            "\t".join(
                [
                    RUN_ID,
                    src,
                    dst,
                    etype,
                    f"{weight:.6f}",
                    json.dumps(channels, ensure_ascii=False),
                ]
            )
            + "\n"
        )

    # --- 2. COPY into Postgres --------------------------------------------

    start = time.perf_counter()

    cur.copy_from(
        StringIO("".join(node_lines)),
        "nexus_scorable",
        columns=("id", "chat_id", "turn_index", "target_type",
                 "text", "domains", "entities", "meta"),
    )

    cur.copy_from(
        StringIO("".join(metrics_lines)),
        "nexus_metrics",
        columns=("scorable_id", "columns", "values", "vector"),
    )

    cur.copy_from(
        StringIO("".join(edge_lines)),
        "nexus_edge",
        columns=("run_id", "src", "dst", "type", "weight", "channels"),
    )

    conn.commit()
    end = time.perf_counter()

    duration = end - start
    total_rows = NUM_NODES + NUM_NODES + NUM_EDGES  # nodes + metrics + edges
    rps = total_rows / duration if duration > 0 else 0.0

    print(f"\nBulk insert completed in {duration:.2f} seconds.")
    print(f"Total rows: {total_rows:,}")
    print(f"Throughput: {rps:,.0f} rows/second\n")

    cur.close()
    return duration, rps


# --------------------------------------------------------------------------- #
# Pulse benchmark                                                             #
# --------------------------------------------------------------------------- #

def run_pulse_traversal_test(conn: PGConnection) -> Tuple[float, float]:
    """
    Simulate NUM_PULSES "mind pulses":
      - Pick random src node
      - Aggregate over neighbors within RUN_ID
      - Weighted sum by coverage(dst)
    """
    if not NODE_IDS:
        print("No nodes loaded; cannot run Pulse benchmark.")
        return 0.0, 0.0

    cur = conn.cursor()
    print(f"Starting Pulse traversal test ({NUM_PULSES:,} pulses)...")

    # Extract a single dimension ("coverage") from the metrics vector
    PULSE_QUERY = """
        SELECT SUM(e.weight * ( (m.vector->>'coverage')::double precision ))
        FROM nexus_edge e
        JOIN nexus_metrics m
          ON e.dst = m.scorable_id
        WHERE e.run_id = %s
          AND e.src   = %s;
    """

    start = time.perf_counter()

    for _ in range(NUM_PULSES):
        src = random.choice(NODE_IDS)
        cur.execute(PULSE_QUERY, (RUN_ID, src))
        _ = cur.fetchone()

    end = time.perf_counter()
    cur.close()

    duration = end - start
    qps = NUM_PULSES / duration if duration > 0 else 0.0

    print(f"\nPulse traversal completed in {duration:.2f} seconds.")
    print(f"Throughput: {qps:,.0f} pulses/second\n")

    return duration, qps


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("--- PostgreSQL Nexus Graph Benchmark ---")
    print(
        f"DB={DB_NAME} host={DB_HOST}:{DB_PORT} | "
        f"nodes={NUM_NODES:,} edges={NUM_EDGES:,} pulses={NUM_PULSES:,}"
    )

    conn = get_db_connection()
    try:
        setup_nexus_tables(conn)
        load_dur, load_rps = run_bulk_graph_construction(conn)
        pulse_dur, pulse_qps = run_pulse_traversal_test(conn)

        print("--- Summary ---")
        print(f"Bulk load: {load_dur:.2f}s, {load_rps:,.0f} rows/s")
        print(f"Pulse:     {pulse_dur:.2f}s, {pulse_qps:,.0f} pulses/s")
    finally:
        conn.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    # Ensure: pip install psycopg2-binary
    # And that DB_NAME exists and is reachable.
    main()
