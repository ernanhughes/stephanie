# stephanie/components/nexus/tools/blossom_graph_builder.py
"""
Blossom Graph Builder
---------------------
Builds a graph (JSON + optional HTML) from a Nexus run directory that contains:
- garden_events.jsonl  (emitted by NexusImproverAsyncAgent._emit_garden_event)
- optionally: baseline graph JSON, blossom_events/*.jsonl, nexus_improver_report.json

Outputs:
- graph_improved.json  (Cytoscape-style nodes/edges with positions)
- garden.html          (pyvis interactive HTML, if pyvis is installed)

Usage:
  python blossom_graph_builder.py --run-dir runs/run-123
  python blossom_graph_builder.py --run-dir runs/run-123 --baseline runs/run-123/baseline.json
  python blossom_graph_builder.py --run-dir runs/run-123 --no-html

Notes:
- No external dependencies required except optional 'pyvis' for the HTML.
- Layout is a tidy layered tree layout based on parent→children edges.
- If there's more than one root, roots are arranged side-by-side.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ----------------------------- IO helpers -----------------------------

def read_jsonl(path: Path) -> List[dict]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # skip malformed lines
                continue
    return out

def read_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# ----------------------------- Graph model -----------------------------

class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[str, dict] = {}
        self.edges: Set[Tuple[str, str, str]] = set()  # (src, dst, etype)
        self.in_deg: Dict[str, int] = defaultdict(int)
        self.out_adj: Dict[str, List[str]] = defaultdict(list)

    def ensure_node(self, nid: str) -> dict:
        if nid not in self.nodes:
            self.nodes[nid] = {"data": {"id": nid}, "position": {"x": 0.0, "y": 0.0}}
        return self.nodes[nid]

    def add_edge(self, src: str, dst: str, etype: str = "blossom") -> None:
        key = (src, dst, etype or "blossom")
        if key not in self.edges:
            self.edges.add(key)
            self.in_deg[dst] += 1
            if src not in self.in_deg:
                self.in_deg[src] = self.in_deg[src]  # touch
            self.out_adj[src].append(dst)
            self.ensure_node(src)
            self.ensure_node(dst)

    def roots(self) -> List[str]:
        # nodes with in_deg == 0
        nids = set(self.nodes.keys())
        for (s, d, _) in self.edges:
            nids.add(s); nids.add(d)
        return [nid for nid in nids if self.in_deg.get(nid, 0) == 0]

# ----------------------------- Layout -----------------------------

def compute_subtree_leaf_counts(g: Graph, root: str) -> Dict[str, int]:
    """Return number of leaves in each node's subtree (used for tidy layout)."""
    counts: Dict[str, int] = {}

    def dfs(n: str) -> int:
        children = g.out_adj.get(n, [])
        if not children:
            counts[n] = 1
            return 1
        s = 0
        for c in children:
            s += dfs(c)
        counts[n] = max(1, s)
        return counts[n]

    dfs(root)
    return counts

def tidy_tree_layout(
    g: Graph,
    roots: List[str],
    level_y: float = 140.0,
    node_x_spacing: float = 160.0,
    root_gap: float = 240.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute positions for all nodes using a simple tidy tree layout.
    Multiple roots are placed side-by-side.
    """
    pos: Dict[str, Tuple[float, float]] = {}

    current_x = 0.0
    for r in roots:
        counts = compute_subtree_leaf_counts(g, r)

        def assign(n: str, depth: int, x_start: float) -> float:
            children = g.out_adj.get(n, [])
            if not children:
                x = x_start
                pos[n] = (x, depth * level_y)
                return x + node_x_spacing
            # center parent above its children span
            x_cursor = x_start
            child_centers: List[float] = []
            for c in children:
                span = counts.get(c, 1) * node_x_spacing
                # recursively assign child subtree
                x_cursor = assign(c, depth + 1, x_cursor)
                child_centers.append(pos[c][0])
            cx = sum(child_centers) / max(1, len(child_centers))
            pos[n] = (cx, depth * level_y)
            return x_cursor

        # place this root
        end_x = assign(r, 0, current_x)
        current_x = end_x + root_gap

    return pos

# ----------------------------- Styling -----------------------------

def pick_node_style(node_meta: dict) -> dict:
    """
    Return a small style dict; consumer can map to colors in HTML.
    """
    style = {"class": []}
    if node_meta.get("is_root"):
        style["class"].append("root")
    if node_meta.get("promoted"):
        style["class"].append("promoted")
    if node_meta.get("winner"):
        style["class"].append("winner")
    return style

# ----------------------------- Build from events -----------------------------

def build_graph_from_events(run_dir: Path, baseline_path: Optional[Path] = None) -> Dict[str, Any]:
    ge_path = run_dir / "garden_events.jsonl"
    events = read_jsonl(ge_path)

    g = Graph()

    # Accumulate nodes/edges + meta from events
    node_meta: Dict[str, dict] = defaultdict(dict)
    winner_ids: Set[str] = set()

    for ev in events:
        kind = ev.get("kind")
        if kind == "episode_start":
            pid = str(ev.get("parent_id") or "")
            if pid:
                g.ensure_node(pid)
                node_meta[pid]["is_root"] = True
                node_meta[pid]["parent_overall"] = float(ev.get("parent_overall", 0.0))
                node_meta[pid]["text_len"] = int(ev.get("parent_text_len", 0) or 0)
        elif kind == "add_node":
            nid = str(ev.get("node_id") or "")
            pid = str(ev.get("parent_id") or "")
            if nid:
                g.ensure_node(nid)
                node_meta[nid]["overall"] = float(ev.get("overall", 0.0))
                node_meta[nid]["dims"] = ev.get("dims") or {}
                node_meta[nid]["text_len"] = int(ev.get("text_len", 0) or 0)
            if pid and nid:
                g.add_edge(pid, nid, etype=str(ev.get("edge_type") or "blossom"))
        elif kind == "add_edge":
            s = str(ev.get("source") or "")
            t = str(ev.get("target") or "")
            if s and t:
                g.add_edge(s, t, etype=str(ev.get("edge_type") or "blossom"))
        elif kind == "promote":
            cid = str(ev.get("child_id") or "")
            if cid:
                node_meta[cid]["promoted"] = True
        elif kind == "decision":
            wid = str(ev.get("winner_id") or "")
            if wid:
                winner_ids.add(wid)

    for wid in winner_ids:
        node_meta[wid]["winner"] = True

    # Layout
    roots = g.roots() or list(g.nodes.keys())
    pos = tidy_tree_layout(g, roots)

    # Blend baseline positions if provided
    base = read_json(baseline_path) if baseline_path else None
    base_pos = {}
    if base:
        try:
            for n in base.get("nodes", []):
                nid = n["data"]["id"]
                p = n.get("position") or {}
                if "x" in p and "y" in p:
                    base_pos[nid] = (float(p["x"]), float(p["y"]))
        except Exception:
            base_pos = {}

    # Compose cytoscape-style graph
    nodes_out: List[dict] = []
    for nid, nd in g.nodes.items():
        x, y = base_pos.get(nid, pos.get(nid, (0.0, 0.0)))
        meta = node_meta.get(nid, {})
        style = pick_node_style(meta)
        data = {"id": nid, **meta, **style}
        nodes_out.append({"data": data, "position": {"x": x, "y": y}})

    edges_out: List[dict] = []
    for (s, t, et) in g.edges:
        eid = f"{s}->{t}"
        edges_out.append({"data": {"id": eid, "source": s, "target": t, "type": et}})

    return {"nodes": nodes_out, "edges": edges_out}

# ----------------------------- HTML (pyvis) -----------------------------

def try_render_pyvis_html(graph: Dict[str, Any], out_html: Path) -> bool:
    try:
        from pyvis.network import Network
    except Exception:
        return False

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=160, spring_strength=0.015)

    # color map
    def color_for(d: dict) -> str:
        classes = set((d.get("class") or []))
        if "promoted" in classes:
            return "#2ca02c"  # green
        if "winner" in classes:
            return "#ff8c00"  # orange
        if "root" in classes:
            return "#4c78a8"  # blue
        return "#9da3a6"  # grey

    for n in graph["nodes"]:
        d = n["data"]
        p = n.get("position", {})
        net.add_node(
            d["id"],
            label=d["id"],
            title=json.dumps({k: v for k, v in d.items() if k not in {"id", "class"}}, ensure_ascii=False, indent=2),
            x=p.get("x"),
            y=p.get("y"),
            color=color_for(d),
            physics=False,   # use our layout positions
        )

    for e in graph["edges"]:
        d = e["data"]
        net.add_edge(d["source"], d["target"], title=d.get("type", "edge"))

    net.set_options("""
        {
        "nodes": {
            "shape": "dot",
            "size": 12,
            "font": { "face": "Inter" }
        },
        "edges": {
            "smooth": false,
            "width": 1
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 120
        },
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "springLength": 120,
            "springConstant": 0.05
            },
            "minVelocity": 0.75
        }
        }
        """)
    # Avoid the notebook template path (which can be None on CLI):
    net.write_html(out_html.as_posix(), open_browser=False, notebook=False)
    return True

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to runs/run-<id> directory")
    ap.add_argument("--baseline", default=None, help="Optional baseline graph JSON with positions")
    ap.add_argument("--no-html", action="store_true", help="Skip HTML render even if pyvis is available")
    ap.add_argument("--out-json", default="graph_improved.json", help="Output JSON filename")
    ap.add_argument("--out-html", default="garden.html", help="Output HTML filename")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"[!] run dir not found: {run_dir}")

    baseline_path = Path(args.baseline) if args.baseline else None
    graph = build_graph_from_events(run_dir, baseline_path)
    out_json = run_dir / args.out_json
    write_json(out_json, graph)
    print(f"[✓] Wrote {out_json}")

    if not args.no_html:
        out_html = run_dir / args.out_html
        if try_render_pyvis_html(graph, out_html):
            print(f"[✓] Wrote {out_html}")
        else:
            print("[i] pyvis not installed; skipped HTML. Install with: pip install pyvis")

if __name__ == "__main__":
    main()
