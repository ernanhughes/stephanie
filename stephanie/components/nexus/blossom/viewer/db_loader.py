# stephanie/components/nexus/viewer/db_loader.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg  # psycopg 3


# If you prefer SQLAlchemy, swap this for your Session and .execute() calls.

@dataclass
class BlossomFrame:
    step: int
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    meta: Dict[str, Any]

class BlossomDBLoader:
    """
    Reconstructs a growth timeline from blossoms + blossom_nodes + blossom_edges.
    Grouping strategy:
      - primary: created_at of blossom_nodes
      - tie-break: order_index
    """
    def __init__(self, dsn: str):
        self.dsn = dsn

    def load_frames(
        self,
        *,
        pipeline_run_id: Optional[str] = None,
        blossom_id: Optional[int] = None,
        bucket_seconds: Optional[int] = None,
    ) -> List[BlossomFrame]:
        assert pipeline_run_id or blossom_id, "Provide pipeline_run_id or blossom_id"
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                if blossom_id is None:
                    cur.execute(
                        """
                        SELECT id, goal_id, pipeline_run_id, strategy, started_at, completed_at
                        FROM blossoms
                        WHERE pipeline_run_id = %s
                        ORDER BY started_at DESC
                        LIMIT 1
                        """,
                        (str(pipeline_run_id),),
                    )
                    row = cur.fetchone()
                    if not row:
                        return []
                    blossom_id = int(row[0])

                # Pull nodes & edges
                cur.execute(
                    """
                    SELECT id, node_id, parent_id, root_node_id, node_type, depth,
                           order_index, plan_text, state_text, sharpened_text,
                           accepted, scores, created_at
                    FROM blossom_nodes
                    WHERE blossom_id = %s
                    ORDER BY created_at ASC, COALESCE(order_index, 0) ASC, id ASC
                    """,
                    (blossom_id,),
                )
                nodes = cur.fetchall()

                cur.execute(
                    """
                    SELECT src_node_id, dst_node_id, relation, score, created_at
                    FROM blossom_edges
                    WHERE blossom_id = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                    (blossom_id,),
                )
                edges = cur.fetchall()

        # Build frames
        frames: List[BlossomFrame] = []
        all_nodes: List[Dict[str, Any]] = []
        all_edges: List[Dict[str, Any]] = []

        def node_row_to_dict(r):
            return {
                "bn_id": r[0],
                "id": r[1],
                "parent_id": r[2],
                "root_id": r[3],
                "type": r[4],
                "depth": r[5],
                "order": r[6],
                "plan_text": r[7],
                "state_text": r[8],
                "sharpened_text": r[9],
                "accepted": bool(r[10]) if r[10] is not None else False,
                "scores": (r[11] or {}),
                "created_at": r[12].isoformat() if isinstance(r[12], dt.datetime) else r[12],
            }

        def edge_row_to_dict(r):
            return {
                "src": r[0],
                "dst": r[1],
                "relation": r[2] or "child",
                "score": float(r[3]) if r[3] is not None else None,
                "created_at": r[4].isoformat() if isinstance(r[4], dt.datetime) else r[4],
            }

        # A simple “one-step-per-node” growth:
        by_step_nodes: List[Dict[str, Any]] = [node_row_to_dict(r) for r in nodes]
        by_step_edges: List[Dict[str, Any]] = [edge_row_to_dict(r) for r in edges]

        # Optionally bucket by time
        if bucket_seconds:
            def bucket(ts):
                t = dt.datetime.fromisoformat(ts.replace("Z","")).timestamp()
                return int(t // bucket_seconds)
            for i, nd in enumerate(by_step_nodes):
                nd["__bucket"] = bucket(nd["created_at"]) if nd.get("created_at") else i
            for i, ed in enumerate(by_step_edges):
                ed["__bucket"] = bucket(ed["created_at"]) if ed.get("created_at") else i

            buckets = sorted(set([nd["__bucket"] for nd in by_step_nodes] + [ed["__bucket"] for ed in by_step_edges]))
            step_map = {b: i for i, b in enumerate(buckets)}
            # Assign step by bucket
            node_steps = [(step_map[nd["__bucket"]], nd) for nd in by_step_nodes]
            edge_steps = [(step_map[ed["__bucket"]], ed) for ed in by_step_edges]
        else:
            # Assign step by insertion order (nodes drive the rhythm)
            node_steps = list(enumerate(by_step_nodes))
            # Edges appear when both ends exist; we add them lazily below.
            edge_steps = [(len(node_steps) - 1, ed) for ed in by_step_edges]

        # Build cumulative frames
        seen_nodes = set()
        for step_idx, nd in node_steps:
            if nd["id"] not in seen_nodes:
                all_nodes.append(nd)
                seen_nodes.add(nd["id"])

            # pull in any edges whose endpoints exist by now
            step_edges_now = []
            for k, ed in list(enumerate(edge_steps)):
                s, e = ed
                if s <= step_idx and e["src"] in seen_nodes and e["dst"] in seen_nodes:
                    all_edges.append(e)
                    step_edges_now.append(k)
            # (optional) remove consumed edges

            frames.append(
                BlossomFrame(
                    step=len(frames),
                    nodes=list(all_nodes),
                    edges=list(all_edges),
                    meta={"blossom_id": blossom_id},
                )
            )
        return frames
