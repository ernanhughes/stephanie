#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nexus/Blossom Growth Viewer
---------------------------

Reads per-episode JSONL event logs from:
    runs/run-<pipeline_run_id>/blossom_events/*.jsonl

Reconstructs an evolving graph (nodes/edges) and renders:
  - A filmstrip PNG showing graph growth across time.
  - Optionally an animated GIF (if imageio installed).

We intentionally avoid DB coupling; this script relies only on the JSONL logs
your NexusImproverAgent already writes. It also (optionally) reads a
nexus_improver_report.json next to those logs, if present, to annotate metrics.

Usage:
    python tools/reports/nexus_growth_report.py \
        --run-root runs/run-1699480000 \
        --out-dir runs/run-1699480000/viz \
        --stride 5 \
        --gif

Notes:
  * Layout is computed once on the final graph and reused for all frames to keep
    node positions stable across the filmstrip.
  * We render a frame every `stride` growth steps (add_node/add_edge). Lower
    stride = more frames.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Minimal deps: networkx + matplotlib, imageio is optional
import networkx as nx
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio  # optional, for GIF
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _collect_events(events_dir: Path) -> List[Dict[str, Any]]:
    """
    Collect events from all episode JSONLs and order them by (ts, file order).
    Expected event shapes (best-effort; we are permissive):
      { "kind": "add_node", "id": int|str, "ts": float|int, ... }
      { "kind": "add_edge", "src": id, "dst": id, "ts": float|int, ... }
      { "kind": "update_node", "id": id, "reward": float, ... }  # optional
    """
    all_paths = sorted(Path(events_dir).glob("*.jsonl"))
    events: List[Tuple[float, int, Dict[str, Any]]] = []
    seq = 0
    for p in all_paths:
        for ev in _iter_jsonl(p):
            ts = ev.get("ts")
            # fallback monotonically increasing order if no timestamp
            if ts is None:
                ts = float(seq)
            events.append((float(ts), seq, ev))
            seq += 1
    events.sort(key=lambda t: (t[0], t[1]))
    return [e[2] for e in events]


def _apply_event(G: nx.Graph, ev: Dict[str, Any]):
    k = ev.get("kind")
    if k == "add_node":
        nid = ev.get("id") or ev.get("node_id") or ev.get("leaf_id")
        if nid is None:
            return
        if not G.has_node(nid):
            G.add_node(nid)
        # persist interesting attributes if present
        for attr in ("reward", "value", "text_len", "episode_id"):
            if attr in ev:
                G.nodes[nid][attr] = ev[attr]
    elif k == "update_node":
        nid = ev.get("id") or ev.get("node_id")
        if nid is None:
            return
        if not G.has_node(nid):
            G.add_node(nid)
        for attr in ("reward", "value", "text_len"):
            if attr in ev:
                G.nodes[nid][attr] = ev[attr]
    elif k == "add_edge":
        u = ev.get("src") or ev.get("u") or ev.get("parent_id")
        v = ev.get("dst") or ev.get("v") or ev.get("child_id")
        if u is None or v is None:
            return
        if not G.has_node(u):
            G.add_node(u)
        if not G.has_node(v):
            G.add_node(v)
        if not G.has_edge(u, v):
            G.add_edge(u, v)


def _build_snapshots(events: List[Dict[str, Any]], stride: int = 5):
    """
    Apply events to a working graph, saving a snapshot after every `stride`
    growth event (node or edge add). Returns:
      - final_graph
      - snapshots: list of sets (nodes_present, edges_present) for each frame
    """
    G = nx.Graph()
    snapshots: List[Tuple[set, set]] = []
    growth_count = 0

    for ev in events:
        before_nodes = set(G.nodes)
        before_edges = set(G.edges)

        _apply_event(G, ev)

        grew = (len(G.nodes) > len(before_nodes)) or (len(G.edges) > len(before_edges))
        if grew:
            growth_count += 1
            if growth_count % max(1, stride) == 0:
                snapshots.append((set(G.nodes), set(G.edges)))

    # Always include a final snapshot
    if not snapshots or snapshots[-1][0] != set(G.nodes) or snapshots[-1][1] != set(G.edges):
        snapshots.append((set(G.nodes), set(G.edges)))

    return G, snapshots


def _read_report(run_root: Path) -> Dict[str, Any]:
    """Optional: read nexus_improver_report.json to annotate metrics."""
    rpt = run_root / "nexus_improver_report.json"
    if rpt.exists():
        try:
            return json.loads(rpt.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _render_frames(final_G: nx.Graph,
                   snapshots: List[Tuple[set, set]],
                   out_dir: Path,
                   report: Dict[str, Any]) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stable layout on final graph (reuse across frames).
    # Seed ensures determinism, and k scales with graph size implicitly.
    pos = nx.spring_layout(final_G, seed=42)

    frame_paths: List[Path] = []
    for i, (nodes_present, edges_present) in enumerate(snapshots, start=1):
        fig = plt.figure(figsize=(10, 8), dpi=120)

        # Subgraph view
        H = nx.Graph()
        H.add_nodes_from(nodes_present)
        H.add_edges_from(e for e in edges_present if e[0] in nodes_present and e[1] in nodes_present)

        # Compute simple node sizes (degree-based) for readability
        deg = dict(H.degree)
        sizes = [300 + 40 * deg.get(n, 0) for n in H.nodes]

        # Draw
        nx.draw_networkx_edges(H, pos, edgelist=list(H.edges), alpha=0.6)
        nx.draw_networkx_nodes(H, pos, nodelist=list(H.nodes), node_size=sizes, alpha=0.9)
        # Optional: label a few highest-degree nodes to reduce clutter
        if len(H) <= 60:
            nx.draw_networkx_labels(H, pos, font_size=8)

        # Title + metrics overlay
        n_nodes, n_edges = H.number_of_nodes(), H.number_of_edges()
        plt.title(f"Nexus/Blossom Growth — Frame {i}/{len(snapshots)} — nodes={n_nodes}, edges={n_edges}")

        # Report overlay if present
        if report:
            try:
                win_rate = report.get("win_rate")
                mean_lift = report.get("mean_lift")
                topk_lift = report.get("topk_lift")
                baseline = report.get("baseline", {})
                txt = [
                    f"win_rate: {win_rate:.2f}" if isinstance(win_rate, (int, float)) else None,
                    f"mean_lift: {mean_lift:.3f}" if isinstance(mean_lift, (int, float)) else None,
                    f"topk_lift: {topk_lift:.3f}" if isinstance(topk_lift, (int, float)) else None,
                    f"baseline mean/median/p90: {baseline.get('mean',0):.3f}/"
                    f"{baseline.get('median',0):.3f}/{baseline.get('p90',0):.3f}",
                ]
                txt = "\n".join([t for t in txt if t])
                if txt:
                    plt.gcf().text(0.02, 0.02, txt, ha="left", va="bottom", fontsize=8)
            except Exception:
                pass

        plt.axis("off")
        frame_path = out_dir / f"frame_{i:04d}.png"
        plt.savefig(frame_path, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    return frame_paths


def _make_filmstrip(frame_paths: List[Path], out_path: Path):
    """Stack frames vertically into a single filmstrip PNG (no extra deps)."""
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    if not imgs:
        return
    w = max(im.width for im in imgs)
    total_h = sum(im.height for im in imgs)
    out = Image.new("RGB", (w, total_h))
    y = 0
    for im in imgs:
        out.paste(im, (0, y))
        y += im.height
    out.save(out_path)


def _make_gif(frame_paths: List[Path], out_path: Path, fps: int = 2):
    if not _HAS_IMAGEIO:
        return
    imgs = [imageio.imread(p) for p in frame_paths]
    if imgs:
        imageio.mimsave(out_path, imgs, duration=1.0 / max(1, fps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="e.g., runs/run-1699480000")
    ap.add_argument("--out-dir", default=None, help="output dir for frames/filmstrip")
    ap.add_argument("--stride", type=int, default=5, help="save a frame every N growth steps")
    ap.add_argument("--gif", action="store_true", help="also write an animated GIF if imageio is available")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    events_dir = run_root / "blossom_events"
    out_dir = Path(args.out_dir or (run_root / "viz"))
    out_dir.mkdir(parents=True, exist_ok=True)

    events = _collect_events(events_dir)
    if not events:
        print(f"[nexus_growth] no events found in {events_dir}")
        return

    final_G, snapshots = _build_snapshots(events, stride=max(1, args.stride))
    report = _read_report(run_root)

    frames = _render_frames(final_G, snapshots, out_dir, report)

    strip_path = out_dir / "filmstrip.png"
    _make_filmstrip(frames, strip_path)
    print(f"[nexus_growth] filmstrip: {strip_path}")

    if args.gif and _HAS_IMAGEIO:
        gif_path = out_dir / "growth.gif"
        _make_gif(frames, gif_path, fps=2)
        print(f"[nexus_growth] gif: {gif_path}")
    elif args.gif:
        print("[nexus_growth] imageio not installed; skipping GIF")


if __name__ == "__main__":
    main()
