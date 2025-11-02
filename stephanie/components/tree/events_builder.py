from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def build_tree_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Accepts a list of event recs like those in TreeEventEmitter.recent().
    Returns:
      {
        "roots": [root_ids...],
        "children": {parent_id: [child_id, ...], ...},
        "nodes": {id: node_dict, ...},  # last-seen snapshot
        "order": [ids...]               # topo-ish order by arrival
      }
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    children = defaultdict(list)
    roots: List[str] = []
    order: List[str] = []

    def _grab(nrec: Optional[Dict[str, Any]]) -> Optional[str]:
        if not nrec: return None
        nid = nrec.get("id")
        if nid:
            nodes[nid] = nrec
        return nid

    for rec in events:
        if rec.get("kind") != "tree_event":
            continue
        e = rec.get("event")
        p = rec.get("payload", {})
        if e == "root_created":
            cid = _grab(p.get("node"))
            if cid is not None and cid not in roots:
                roots.append(cid)
                order.append(cid)
        elif e == "node_added":
            pid = _grab(p.get("parent"))
            cid = _grab(p.get("child"))
            if pid is not None and cid is not None:
                children[pid].append(cid)
                order.append(cid)
        elif e == "best_update":
            # you could mark nodes[nid]["is_best"]=True if desired
            nid = _grab(p.get("node"))
            if nid:
                nodes[nid]["is_best"] = True
        # other events (expand/backprop/progress) don’t change structure

    return {"roots": roots, "children": dict(children), "nodes": nodes, "order": order}
