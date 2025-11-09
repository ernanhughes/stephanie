#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nexus/Blossom Growth Viewer (DB-backed)
---------------------------------------

Builds an evolving graph from your Postgres tables:

    blossoms(id, goal_id, pipeline_run_id, ...)
    blossom_nodes(id, blossom_id, node_id, parent_id, scores, accepted, created_at, ...)
    blossom_edges(id, blossom_id, src_node_id, dst_node_id, created_at, ...)

It reconstructs graph growth ordered by *created_at*, snapshots every N growth
steps, and renders:
  - frames:   PNGs per snapshot
  - filmstrip.png: vertical stack of frames
  - growth.gif: animated (optional, requires imageio)

Usage examples:
    # by pipeline run id (coalesces all episodes in that run)
    python tools/reports/nexus_growth_from_db.py \
      --dsn "postgresql://user:pass@localhost:5432/stephanie" \
      --pipeline-run-id run-1731080000 \
      --out-dir runs/run-1731080000/viz --stride 5 --gif

    # by a single episode
    python tools/reports/nexus_growth_from_db.py \
      --dsn $DATABASE_URL \
      --blossom-id 1234 --out-dir runs/episode-1234/viz

Performance notes:
  • Add these indexes if you haven’t:
      CREATE INDEX IF NOT EXISTS idx_bn_created  ON blossom_nodes(created_at);
      CREATE INDEX IF NOT EXISTS idx_be_created  ON blossom_edges(created_at);
  • We fetch only the rows for your run/episode filter.

Frame semantics:
  • A new frame is captured every `stride` growth events (node add OR edge add).
  • Layout is computed once on the final graph for positional stability.
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psycopg  # pip install psycopg[binary]
import networkx as nx
import matplotlib.pyplot as plt

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    import imageio.v2 as imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False


# ------------------------- DB fetch -------------------------

def _fetch_ids(conn, pipeline_run_id: Optional[str], blossom_id: Optional[int]) -> List[int]:
    q, params = None, None
    if blossom_id is not None:
        q = "SELECT id FROM blossoms WHERE id = %s"
        params = (int(blossom_id),)
    elif pipeline_run_id:
        q = "SELECT id FROM blossoms WHERE pipeline_run_id = %s ORDER BY id"
        params = (pipeline_run_id,)
    else:
        raise ValueError("Provide --pipeline-run-id or --blossom-id")
    with conn.cursor() as cur:
        cur.execute(q, params)
        return [r[0] for r in cur.fetchall()]

def _fetch_nodes(conn, blossom_ids: Iterable[int]) -> List[Dict[str, Any]]:
    if not blossom_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT blossom_id, node_id, parent_id, root_node_id, node_type,
                   depth, order_index, accepted,
                   scores, tags, features,
                   sharpened_text, state_text, plan_text,
                   created_at
            FROM blossom_nodes
            WHERE blossom_id = ANY(%s)
            ORDER BY created_at ASC, id ASC
            """,
            (list(blossom_ids),),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

def _fetch_edges(conn, blossom_ids: Iterable[int]) -> List[Dict[str, Any]]:
    if not blossom_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT blossom_id, src_node_id, dst_node_id,
                   relation, score, rationale, created_at
            FROM blossom_edges
            WHERE blossom_id = ANY(%s)
            ORDER BY created_at ASC, id ASC
            """,
            (list(blossom_ids),),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

def _json_score_val(scores: Any, key: str, default: Optional[float]=None) -> Optional[float]:
    if not scores:
        return default
    if isinstance(scores, dict):
        v = scores.get(key, default)
        try:
            return float(v) if v is not None else default
        except Exception:
            return default
    # if DB driver returns as str
    try:
        d = json.loads(scores)
        v = d.get(key, default)
        return float(v) if v is not None else default
    except Exception:
        return default


# ------------------------- Build events -------------------------

def _build_events(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Turn rows into a single ordered stream of growth events:
      ("node", created_at, payload) or ("edge", created_at, payload)
    Returns list sorted by timestamp (float epoch seconds); ties keep ordering.
    """
    evs: List[Tuple[float, Dict[str, Any]]] = []

    def _ts(dt) -> float:
        # psycopg returns datetime; convert to epoch-ish float for sorting stability
        try:
            return dt.timestamp()
        except Exception:
            return 0.0

    for n in nodes:
        evs.append((
            _ts(n["created_at"]),
            {
                "kind": "add_node",
                "node_id": n["node_id"],
                "parent_id": n["parent_id"],
                "accepted": bool(n.get("accepted", False)),
                "depth": n.get("depth"),
                "score_overall": _json_score_val(n.get("scores"), "overall", None),
                "score_reward": _json_score_val(n.get("scores"), "reward", None),
            },
        ))

    for e in edges:
        evs.append((
            _ts(e["created_at"]),
            {
                "kind": "add_edge",
                "src": e["src_node_id"],
                "dst": e["dst_node_id"],
                "relation": e.get("relation", "child") or "child",
                "score": e.get("score"),
            },
        ))

    # order by ts; stable within equal ts
    evs.sort(key=lambda t: t[0])
    return evs


# ------------------------- Graph snapshots -------------------------

def _apply_event(G: nx.Graph, ev: Dict[str, Any]) -> bool:
    """Apply to G; return True if graph grew (node/edge added)."""
    k = ev.get("kind")
    if k == "add_node":
        nid = ev.get("node_id")
        if not nid:
            return False
        grew = False
        if not G.has_node(nid):
            G.add_node(nid)
            grew = True
        # set/update useful attrs
        G.nodes[nid]["accepted"] = bool(ev.get("accepted", False))
        if ev.get("depth") is not None:
            G.nodes[nid]["depth"] = int(ev["depth"])
        if ev.get("score_overall") is not None:
            G.nodes[nid]["overall"] = float(ev["score_overall"])
        if ev.get("score_reward") is not None:
            G.nodes[nid]["reward"] = float(ev["score_reward"])
        # implicit parent edge if present and not already in edges table
        pid = ev.get("parent_id")
        if pid:
            if not G.has_node(pid):
                G.add_node(pid)
            if not G.has_edge(pid, nid):
                G.add_edge(pid, nid)
                grew = True
        return grew

    elif k == "add_edge":
        u, v = ev.get("src"), ev.get("dst")
        if not u or not v:
            return False
        grew = False
        if not G.has_node(u):
            G.add_node(u)
            grew = True
        if not G.has_node(v):
            G.add_node(v)
            grew = True
        if not G.has_edge(u, v):
            G.add_edge(u, v, relation=ev.get("relation", "child"))
            grew = True
        return grew

    return False


def _build_snapshots(events: List[Tuple[float, Dict[str, Any]]], stride: int) -> Tuple[nx.Graph, List[Tuple[set, set]]]:
    G = nx.Graph()
    snapshots: List[Tuple[set, set]] = []
    growth = 0

    for _, ev in events:
        grew = _apply_event(G, ev)
        if grew:
            growth += 1
            if growth % max(1, stride) == 0:
                snapshots.append((set(G.nodes), set(G.edges)))

    if not snapshots or snapshots[-1][0] != set(G.nodes) or snapshots[-1][1] != set(G.edges):
        snapshots.append((set(G.nodes), set(G.edges)))
    return G, snapshots


# ------------------------- Rendering -------------------------

def _node_colors(G: nx.Graph, nodes: List[Any]) -> List[Tuple[float, float, float]]:
    """
    Light grey for regular, darker/greenish for accepted.
    We avoid external colormap deps and keep it readable.
    """
    colors = []
    for n in nodes:
        acc = bool(G.nodes[n].get("accepted", False))
        if acc:
            colors.append((0.25, 0.55, 0.35))  # muted green
        else:
            colors.append((0.65, 0.65, 0.70))  # grey-blue
    return colors

def _node_sizes(G: nx.Graph, nodes: List[Any]) -> List[int]:
    deg = dict(G.degree)
    return [300 + 60 * deg.get(n, 0) for n in nodes]

def _render_frames(final_G: nx.Graph, snapshots: List[Tuple[set, set]], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(final_G, seed=42)  # stable across frames
    frame_paths: List[Path] = []

    for i, (Nset, Eset) in enumerate(snapshots, start=1):
        fig = plt.figure(figsize=(10, 8), dpi=120)

        H = nx.Graph()
        H.add_nodes_from(Nset)
        H.add_edges_from([e for e in Eset if e[0] in Nset and e[1] in Nset])

        nodelist = list(H.nodes)
        edgelist = list(H.edges)

        nx.draw_networkx_edges(H, pos, edgelist=edgelist, alpha=0.55)
        nx.draw_networkx_nodes(
            H, pos,
            nodelist=nodelist,
            node_color=_node_colors(H, nodelist),
            node_size=_node_sizes(H, nodelist),
            alpha=0.95
        )

        # Labels only on small-ish graphs
        if H.number_of_nodes() <= 60:
            nx.draw_networkx_labels(H, pos, font_size=8)

        # Overlay
        plt.title(f"Blossom Growth — Frame {i}/{len(snapshots)} — nodes={H.number_of_nodes()} edges={H.number_of_edges()}")
        plt.axis("off")

        fp = out_dir / f"frame_{i:04d}.png"
        plt.savefig(fp, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(fp)
    return frame_paths

def _make_filmstrip(frame_paths: List[Path], out_path: Path):
    if not _HAS_PIL or not frame_paths:
        return
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    w = max(im.width for im in imgs)
    total_h = sum(im.height for im in imgs)
    out = Image.new("RGB", (w, total_h))
    y = 0
    for im in imgs:
        out.paste(im, (0, y))
        y += im.height
    out.save(out_path)

def _make_gif(frame_paths: List[Path], out_path: Path, fps: int = 2):
    if not _HAS_IMAGEIO or not frame_paths:
        return
    imgs = [imageio.imread(p) for p in frame_paths]
    if imgs:
        imageio.mimsave(out_path, imgs, duration=1.0 / max(1, fps))


# ------------------------- CLI -------------------------

# python .\stephanie\reporting\nexus_growth_from_db.py --dsn "postgresql://co:co@localhost:5432/co" --pipeline-run-id 8246

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.getenv("DATABASE_URL"), help="Postgres DSN, e.g., postgresql://user:pass@host/db")
    ap.add_argument("--pipeline-run-id", default=None, help="Aggregate all blossoms with this pipeline_run_id")
    ap.add_argument("--blossom-id", type=int, default=None, help="Single blossom/episode id")
    ap.add_argument("--out-dir", default="runs/viz", help="Output directory for frames/filmstrip.gif")
    ap.add_argument("--stride", type=int, default=5, help="Capture a frame every N growth events")
    ap.add_argument("--gif", action="store_true", help="Also write growth.gif (requires imageio)")
    args = ap.parse_args()

    if not args.dsn:
        raise SystemExit("Provide --dsn or set DATABASE_URL")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(args.dsn) as conn:
        blossom_ids = _fetch_ids(conn, args.pipeline_run_id, args.blossom_id)
        if not blossom_ids:
            print("[growth] No blossoms matched the filter.")
            return
        nodes = _fetch_nodes(conn, blossom_ids)
        edges = _fetch_edges(conn, blossom_ids)

    events = _build_events(nodes, edges)
    if not events:
        print("[growth] No events (nodes/edges) found.")
        return

    final_G, snapshots = _build_snapshots(events, stride=max(1, args.stride))
    frames = _render_frames(final_G, snapshots, out_dir)

    film = out_dir / "filmstrip.png"
    _make_filmstrip(frames, film)
    print(f"[growth] filmstrip: {film}")

    if args.gif:
        gif = out_dir / "growth.gif"
        _make_gif(frames, gif, fps=2)
        if gif.exists():
            print(f"[growth] gif: {gif}")
        else:
            print("[growth] imageio not available; skipped GIF")

if __name__ == "__main__":
    main()
