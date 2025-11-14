# stephanie/tools/build_blossom_film.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------
# IO helpers
# ---------------------------

def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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
                # best-effort
                continue
    return out

# ---------------------------
# Film builder
# ---------------------------

def build_frames(
    *,
    events_path: Path,
    baseline_graph_json: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[float, float]]]:
    """
    Build incremental frames from garden_events.jsonl (+ optional baseline graph with positions).

    Returns:
      frames: [{
        "nodes": [{"data": {"id": ...,"cls": "baseline"|"new"}, "position": {"x":..,"y":..}}],
        "edges": [{"data": {"id": "...", "source": "...", "target": "..."}}],
        "event": "add_node|add_edge|decision|promote|episode_start|episode_end|...",
        "ts": <int>,
        "highlight": {"node_id": str, "edge": (src, dst), "type": "add|promote|decision"}
      }, ...]
      base_pos: {node_id: (x,y)}
    """
    # ---- 1) baseline positions
    base_pos: Dict[str, Tuple[float, float]] = {}
    base_nodes = set()
    base_edges = set()

    g = _load_json(baseline_graph_json)
    if g:
        for n in g.get("nodes", []):
            nid = str(n["data"]["id"])
            base_nodes.add(nid)
            pos = n.get("position") or {}
            if isinstance(pos.get("x"), (int, float)) and isinstance(pos.get("y"), (int, float)):
                base_pos[nid] = (float(pos["x"]), float(pos["y"]))
        for e in g.get("edges", []):
            src = str(e["data"]["source"])
            dst = str(e["data"]["target"])
            base_edges.add((src, dst))

    # ---- 2) collect events
    events: List[Dict[str, Any]] = _load_jsonl(events_path)

    # ---- 3) garden state (incremental)
    nodes: Dict[str, Dict[str, Any]] = {}
    edges = set(base_edges)
    frames: List[Dict[str, Any]] = []

    # seed baseline nodes into state (class=baseline)
    for nid in base_nodes:
        nodes[nid] = {
            "data": {"id": nid, "cls": "baseline"},
            "position": {"x": base_pos.get(nid, (0.0, 0.0))[0], "y": base_pos.get(nid, (0.0, 0.0))[1]},
        }

    # placement helpers
    anchors_used = 0
    def anchor_for_root(i: int, R: float = 420.0, per_ring: int = 12) -> Tuple[float, float]:
        # place new roots on expanding rings
        ring = i // per_ring
        k = i % per_ring
        theta = 2 * math.pi * (k / per_ring)
        r = (1 + ring) * R
        return (r * math.cos(theta), r * math.sin(theta))

    def ensure_node(nid: Optional[str], cls: str = "new") -> str:
        """Ensure node exists; if missing position, anchor it."""
        nonlocal anchors_used
        if not nid or nid == "None":
            # synthesize a root id
            nid = f"root:{anchors_used}"
        nid = str(nid)
        if nid not in nodes:
            # position: baseline if known else anchor
            if nid not in base_pos:
                base_pos[nid] = anchor_for_root(anchors_used)
                anchors_used += 1
            nodes[nid] = {
                "data": {"id": nid, "cls": cls},
                "position": {"x": base_pos[nid][0], "y": base_pos[nid][1]},
            }
        return nid

    def place_children_radial(pid: str, child_ids: List[str], base_r: float = 90.0):
        """Immediate radial layout around parent."""
        px, py = base_pos.get(pid, (0.0, 0.0))
        n = max(1, len(child_ids))
        for k, cid in enumerate(child_ids):
            theta = 2 * math.pi * (k / n)
            x = px + base_r * math.cos(theta)
            y = py + base_r * math.sin(theta)
            nodes[cid] = {
                "data": {"id": cid, "cls": "new"},
                "position": {"x": x, "y": y},
            }
            base_pos[cid] = (x, y)

    # Track per-parent pending children so we can “bloom” them together if needed
    pending_children: Dict[str, List[str]] = {}

    # ---- 4) roll events → frames
    for ev in events:
        kind = ev.get("kind")
        hl: Dict[str, Any] = {}

        if kind == "episode_start":
            # treat parent as root anchor if not in baseline
            pid = ensure_node(str(ev.get("parent_id")), cls="new")
            # snapshot frame
        elif kind == "add_node":
            pid = ensure_node(str(ev.get("parent_id")), cls="new")
            cid = ensure_node(str(ev.get("node_id")), cls="new")
            # create edge
            edges.add((pid, cid))
            # place child immediately (gives the “bloom” feel)
            place_children_radial(pid, [cid], base_r=100.0)
            hl = {"node_id": cid, "edge": (pid, cid), "type": "add"}
        elif kind == "add_edge":
            s = ensure_node(str(ev.get("source")), cls="new")
            t = ensure_node(str(ev.get("target")), cls="new")
            edges.add((s, t))
            hl = {"edge": (s, t), "type": "add"}
        elif kind in ("decision", "promote"):
            pid = ensure_node(str(ev.get("parent_id")), cls="new")
            cid = ensure_node(str(ev.get("child_id")), cls="new")
            edges.add((pid, cid))
            # emphasize winner edge & node
            hl = {"node_id": cid, "edge": (pid, cid), "type": kind}
        elif kind == "episode_end":
            # nothing special; snapshot for completeness
            pass
        else:
            # Unknown or informative events → just snapshot
            pass

        # snapshot frame
        frame_nodes = [
            {"data": {"id": nid, "cls": nd["data"].get("cls", "new")}, "position": nd.get("position")}
            for nid, nd in nodes.items()
        ]
        frame_edges = [
            {"data": {"id": f"{s}->{t}", "source": s, "target": t}}
            for (s, t) in sorted(edges)
        ]
        frames.append({
            "nodes": frame_nodes,
            "edges": frame_edges,
            "event": kind,
            "ts": ev.get("ts"),
            "highlight": hl,
        })

    # If no frames (empty events), emit a single baseline frame
    if not frames:
        frame_nodes = [
            {"data": {"id": nid, "cls": "baseline"}, "position": {"x": base_pos.get(nid, (0.0, 0.0))[0],
                                                                  "y": base_pos.get(nid, (0.0, 0.0))[1]}}
            for nid in sorted(base_nodes)
        ]
        frame_edges = [{"data": {"id": f"{s}->{t}", "source": s, "target": t}} for (s, t) in sorted(base_edges)]
        frames.append({"nodes": frame_nodes, "edges": frame_edges, "event": "baseline", "ts": None, "highlight": {}})

    return frames, base_pos

# ---------------------------
# HTML template
# ---------------------------

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Blossom Garden Film</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<style>
  :root {
    --bg:#0e0e0e; --fg:#e5e7eb;
    --base:#6b7280;   /* baseline nodes */
    --new:#97c2fc;    /* new nodes */
    --hi:#f87171;     /* highlight add/promote/decision */
    --edge:#8aa2b2;   /* default edges */
  }
  body { margin:0; background:var(--bg); color:var(--fg); font-family:ui-sans-serif, system-ui;}
  #bar { display:flex; gap:12px; align-items:center; padding:10px 12px; border-bottom:1px solid #222;}
  #mynetwork { width:100%; height: calc(100vh - 60px); }
  .btn { background:#222; color:var(--fg); border:1px solid #444; padding:6px 10px; border-radius:8px; cursor:pointer;}
  .btn:active { transform: translateY(1px); }
  .pill { padding:4px 8px; background:#222; border:1px solid #444; border-radius:999px; margin-left:8px;}
  input[type=range] { width:220px;}
</style>
</head>
<body>
<div id="bar">
  <button id="play" class="btn">▶ Play</button>
  <button id="pause" class="btn">⏸ Pause</button>
  <button id="prev" class="btn">⟨</button>
  <input id="seek" type="range" min="0" max="0" value="0" step="1"/>
  <button id="next" class="btn">⟩</button>
  <label class="pill">Speed <input id="speed" type="range" min="0.1" max="3" value="1" step="0.1"></label>
  <span id="label" class="pill">Frame 0/0</span>
</div>
<div id="mynetwork"></div>
<script>
const FRAMES = __FRAMES__;
const HAS_POSITIONS = __HAS_POSITIONS__;

// Use literal colors (canvas-safe)
const COLORS = {
  base: '#6B7280',   // baseline nodes (gray-500)
  newly: '#60A5FA',  // new blossom nodes (blue-400)
  hi:   '#F59E0B',   // highlight (amber-500)
  edge: '#FFFFFF'    // edges/arrows (white)
};
let container = document.getElementById('mynetwork');
let nodes = new vis.DataSet([]);
let edges = new vis.DataSet([]);
let data = {nodes, edges};
let options = {
  physics: { enabled: !HAS_POSITIONS, solver: 'forceAtlas2Based', stabilization: { iterations: 150 } },
  interaction: { hover: true, zoomView: true, dragView: true },
  nodes: { shape: 'dot', scaling: {min:4, max:20}, font: { color: '#e5e7eb' }},
  edges: { arrows: { to: {enabled: true, scaleFactor: 0.45} }, color:{inherit:false}, smooth:{type:'dynamic'} }
};
let network = new vis.Network(container, data, options);
if (options.physics.enabled) {
  network.once('stabilizationIterationsDone', () => network.stopSimulation());
  network.stabilize(200);
}

// --- frame application
function applyFrame(i){
  const f = FRAMES[i];
  if(!f) return;

  const hiNode = f.highlight?.node_id || null;
  const hiType = f.highlight?.type || null;

  // nodes
  const ns = f.nodes.map(n => {
    const id  = n.data.id;
    const cls = n.data.cls || 'new';
    const isHi = (id === (f.highlight?.node_id || null));

    let color = (cls === 'baseline') ? COLORS.base : COLORS.newly;
    if (isHi) color = COLORS.hi;

    return {
      id,
      label: '', // keep dots clean; hover shows id in title
      title: id,
      color,
      size: isHi ? 18 : (cls === 'baseline' ? 7 : 10),
      x: n.position?.x,
      y: n.position?.y,
      fixed: Number.isFinite(n.position?.x) && Number.isFinite(n.position?.y)
    };
  });

// edges
let hiEdge = f.highlight?.edge;
const es = f.edges.map(e => {
  const s = e.data.source, t = e.data.target;
  const isHi = !!(hiEdge && s === hiEdge[0] && t === hiEdge[1]);
  return {
    id: e.data.id,
    from: s,
    to: t,
    color: isHi ? COLORS.hi : COLORS.edge,
    width: isHi ? 2.2 : 1.6,
    arrows: 'to'
  };
});

  nodes.update(ns);
  edges.update(es);

  document.getElementById('label').textContent = `Frame ${i+1}/${FRAMES.length} • ${f.event || 'event'}`;
  document.getElementById('seek').value = i;
}

let idx = 0;
let playing = false;
let speed = 1.0;
let timer = null;

function step(){
  applyFrame(idx);
  idx = Math.min(idx + 1, FRAMES.length);
  if (idx >= FRAMES.length){ pause(); }
}
function play(){
  if (playing) return;
  playing = true;
  const tick = () => {
    step();
    if (playing) timer = setTimeout(tick, 600 / speed);
  };
  tick();
}
function pause(){ playing = false; if (timer) clearTimeout(timer); }
function seekTo(v){
  const i = Math.max(0, Math.min(FRAMES.length-1, parseInt(v)));
  idx = i;
  applyFrame(idx);
}

document.getElementById('play').onclick = play;
document.getElementById('pause').onclick = pause;
document.getElementById('prev').onclick = () => seekTo(idx-1);
document.getElementById('next').onclick = () => seekTo(idx+1);
document.getElementById('seek').max = Math.max(0, FRAMES.length-1);
document.getElementById('seek').oninput = (e)=>seekTo(e.target.value);
document.getElementById('speed').oninput = (e)=>{ speed = parseFloat(e.target.value || '1'); }

if (FRAMES.length) { applyFrame(0); }
</script>
</body>
</html>
"""

def write_html(path: Path, frames: List[Dict[str, Any]]):
    # Detect whether we have positions in the first frame
    has_positions = False
    if frames:
        for n in frames[0].get("nodes", []):
            pos = n.get("position") or {}
            if isinstance(pos.get("x"), (int, float)) and isinstance(pos.get("y"), (int, float)):
                has_positions = True
                break

    html = (HTML_TEMPLATE
            .replace("__HAS_POSITIONS__", "true" if has_positions else "false")
            .replace("__FRAMES__", json.dumps(frames, ensure_ascii=False)))
    path.write_text(html, encoding="utf-8")

# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Build a blossom film from garden_events.jsonl (and optional baseline graph).")
    p.add_argument("--run", required=True, help="Path to run dir (contains garden_events.jsonl)")
    p.add_argument("--baseline", default="", help="Optional path to baseline graph JSON with positions")
    p.add_argument("--out", default="blossom_film.html", help="Output HTML filename (inside --run)")
    args = p.parse_args()

    run_dir = Path(args.run)
    events_path = run_dir / "garden_events.jsonl"
    if not events_path.exists():
        raise SystemExit(f"Missing events file: {events_path}")

    baseline = Path(args.baseline) if args.baseline else None
    frames, _ = build_frames(events_path=events_path, baseline_graph_json=baseline)

    out_html = run_dir / args.out
    write_html(out_html, frames)
    print(out_html.as_posix())

    # also write a static final snapshot
    if frames:
        final = frames[-1]
        final_html = run_dir / "final_graph.html"
        snap = HTML_TEMPLATE \
            .replace("__HAS_POSITIONS__", "true") \
            .replace("__FRAMES__", json.dumps([final], ensure_ascii=False))
        final_html.write_text(snap, encoding="utf-8")
        print(final_html.as_posix())

if __name__ == "__main__":
    main()
