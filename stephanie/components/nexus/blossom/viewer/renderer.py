import json
import math
from pathlib import Path
from typing import Optional

def write_garden_frames(run_dir: Path, *, baseline_graph_json: Optional[str] = None):
    # 1) load baseline nodes/positions if present
    base_pos = {}
    base_nodes = set()
    base_edges = set()
    if baseline_graph_json and Path(baseline_graph_json).exists():
        g = json.loads(Path(baseline_graph_json).read_text(encoding="utf-8"))
        for n in g.get("nodes", []):
            nid = n["data"]["id"]
            base_nodes.add(nid)
            pos = (n.get("position") or {})
            if "x" in pos and "y" in pos:
                base_pos[nid] = (pos["x"], pos["y"])
        for e in g.get("edges", []):
            base_edges.add((e["data"]["source"], e["data"]["target"]))

    # 2) read garden events
    evs = []
    ge = run_dir / "garden_events.jsonl"
    if ge.exists():
        with ge.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    evs.append(json.loads(line))
                except Exception:
                    continue

    # 3) incremental garden state
    nodes = {nid: {"id": nid, "position": {"x": base_pos.get(nid, (0, 0))[0],
                                        "y": base_pos.get(nid, (0, 0))[1]}}
            for nid in base_nodes}
    edges = set(base_edges)
    frames = []

    # radial placement helper
    def place_children_radial(pid, child_ids, base_r=90.0):
        px, py = base_pos.get(pid, (0.0, 0.0))
        n = max(1, len(child_ids))
        for k, cid in enumerate(child_ids):
            theta = 2 * math.pi * (k / n)
            r = base_r
            nodes[cid] = {"id": cid, "position": {"x": px + r * math.cos(theta),
                                                "y": py + r * math.sin(theta)}}

    pending_children = {}
    evs.sort(key=lambda e: (e.get("ts", 0), e.get("kind", "")))
    for ev in evs:
        kind = ev.get("kind")
        if kind == "add_node":
            pid = ev.get("parent_id"); cid = ev.get("node_id")
            pending_children.setdefault(pid, []).append(cid)
            edges.add((pid, cid))
        elif kind == "add_edge":
            s, t = ev.get("source"), ev.get("target")
            if s and t:
                edges.add((s, t))
        elif kind in ("episode_end", "decision", "promote"):
            # place any pending children when episode closes
            pid = ev.get("parent_id")
            if pid and pid in pending_children:
                place_children_radial(pid, pending_children.pop(pid))
        # snapshot a frame
        frames.append({
            "nodes": [{"data": {"id": nid}, "position": nd.get("position")}
                    for nid, nd in nodes.items()],
            "edges": [{"data": {"id": f"{s}->{t}", "source": s, "target": t}} for (s, t) in edges],
            "metrics": {"event": kind, "t": ev.get("ts")}
        })

    # 4) write artifacts
    (run_dir / "garden_frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
    graph = {
        "nodes": [{"data": {"id": nid}, "position": nd.get("position")} for nid, nd in nodes.items()],
        "edges": [{"data": {"id": f"{s}->{t}", "source": s, "target": t}} for (s, t) in edges],
    }
    (run_dir / "graph_improved.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")

    # 5) optional HTML (if your PyVis exporter is available)
    try:
        from stephanie.components.nexus.graph.exporters import export_pyvis_html
        export_pyvis_html(output_path=(run_dir / "garden.html").as_posix(),
                        nodes=graph["nodes"], edges=graph["edges"],
                        title="Nexus Garden — Blossoms")
    except Exception:
        pass
