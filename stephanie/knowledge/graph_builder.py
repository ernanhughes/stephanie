# graph_builder.py — fold KnowledgeBus events → graph state + embeddings
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


class GraphState:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}  # id -> {type, label, t, ...}
        self.edges: List[Dict[str, Any]] = []       # {src, dst, type, t, w}
        self.last_t: float = 0.0

    def upsert_node(self, nid: str, **attrs):
        n = self.nodes.get(nid, {"id": nid})
        n.update(attrs)
        self.nodes[nid] = n

    def add_edge(self, src: str, dst: str, etype: str, t: float, w: float = 1.0, meta: Dict[str, Any] | None = None):
        self.edges.append({"src": src, "dst": dst, "type": etype, "t": t, "w": w, "meta": meta or {}})

class GraphBuilder:
    def __init__(self, events_path: str = "./runs/knowledge/events.jsonl", snapshot_path: str = "./runs/knowledge/snapshot.json"):
        self.events_path = Path(events_path)
        self.snapshot_path = Path(snapshot_path)
        self.g = GraphState()
        self._pos: Dict[str, Tuple[float,float,float]] = {}  # layout cache

    def _safe_readlines(self) -> List[Dict[str, Any]]:
        if not self.events_path.exists(): return []
        lines = []
        with self.events_path.open() as f:
            for ln in f:
                try: lines.append(json.loads(ln))
                except: continue
        return lines

    def rebuild(self):
        self.g = GraphState()
        for evt in self._safe_readlines():
            self._ingest(evt)
        self._layout()
        self._snapshot()

    def tail_and_update_forever(self, interval=1.0):
        last_n = 0
        while True:
            rows = self._safe_readlines()
            if len(rows) > last_n:
                for evt in rows[last_n:]:
                    self._ingest(evt)
                self._layout(incremental=True)
                self._snapshot()
                last_n = len(rows)
            time.sleep(interval)

    # -------- event folding --------

    def _ingest(self, evt: Dict[str, Any]):
        kind = evt.get("kind"); p = evt.get("payload", {}); t = evt.get("ts", time.time())
        self.g.last_t = max(self.g.last_t, t)

        if kind == "casebook.created":
            cb = p["casebook"]
            self.g.upsert_node(f"cb:{cb['name']}", type="casebook", label=cb["name"], t=t, meta=cb)

        elif kind == "case.added":
            cb_name = p["casebook"]; cid = p["case_id"]
            self.g.upsert_node(f"case:{cid}", type="case", label=cid[:6], t=t, meta=p)
            self.g.add_edge(f"cb:{cb_name}", f"case:{cid}", "contains", t)

        elif kind == "scorable.added":
            case_id = p["case_id"]; role = p["role"]
            sid = p.get("scorable_id") or f"{case_id}:{role}:{int(t)}"
            label = role
            self.g.upsert_node(f"sc:{sid}", type="scorable", label=label, t=t, meta=p)
            self.g.add_edge(f"case:{case_id}", f"sc:{sid}", "emits", t)

            # Special handling: VPM → attach numeric dims as node attributes
            if role == "vpm":
                try:
                    vpm = json.loads(p["text"]) if isinstance(p.get("text"), str) else (p.get("meta") or {})
                except Exception:
                    vpm = p.get("meta") or {}
                score = vpm.get("tests_pass_rate") or vpm.get("correctness") or vpm.get("coverage") or 0.0
                self.g.nodes[f"sc:{sid}"]["score"] = float(score)

        elif kind == "trajectory.step":
            casebook = p["casebook"]; case_id = p["case_id"]
            self.g.add_edge(f"cb:{casebook}", f"case:{case_id}", "step", t, w=0.2, meta={"vpm": p.get("vpm")})
            # optional: chain cases in time order
            # you can store last_case per casebook to link sequentially if desired

        elif kind == "decision.emitted":
            unit = p["unit"]; sig = p["signal"]
            nid = f"dec:{unit}:{int(t)}"
            self.g.upsert_node(nid, type="decision", label=sig, t=t, meta=p)
            # if unit maps to a known case/scorable, wire edge
            # simplest: link to latest case node
            latest_case = self._latest_case()
            if latest_case:
                self.g.add_edge(nid, latest_case, sig.lower(), t, w=1.0, meta=p)

        # You can add more: entities, goals, bandit updates, etc.

    def _latest_case(self) -> str | None:
        cases = [k for k,v in self.g.nodes.items() if v.get("type")=="case"]
        if not cases: return None
        return max(cases, key=lambda k: self.g.nodes[k]["t"])

    # -------- layout (3D force-lite) --------

    def _layout(self, incremental: bool = False):
        # simple radial-by-type + jitter; upgrade to force-3D later
        type_r = {"casebook": 20, "case": 35, "scorable": 55, "decision": 25}
        for nid, n in self.g.nodes.items():
            if nid in self._pos and incremental: 
                continue
            r = type_r.get(n.get("type","scorable"), 60)
            ang = hash(nid) % 360 * math.pi/180.0
            x = r * math.cos(ang)
            y = (hash("y"+nid) % 60) - 30
            z = r * math.sin(ang)
            self._pos[nid] = (x,y,z)
            n["pos"] = {"x": x, "y": y, "z": z}

    def _snapshot(self):
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        out = {"t": self.g.last_t, "nodes": list(self.g.nodes.values()), "edges": self.g.edges}
        self.snapshot_path.write_text(json.dumps(out, indent=2))
