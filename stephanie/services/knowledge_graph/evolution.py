# stephanie/services/knowledge_graph/evolution.py
from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _hash_query(q: str) -> str:
    qn = " ".join((q or "").lower().split())
    return hashlib.sha256(qn.encode("utf-8")).hexdigest()[:16]


def _edge_key(e: Dict[str, Any]) -> Tuple[str, str, str]:
    # accept both {src,dst} and {source,target}
    src = e.get("src") or e.get("source") or e.get("source_id")
    dst = e.get("dst") or e.get("target") or e.get("target_id")
    typ = e.get("type") or e.get("rel_type")
    return (str(src), str(dst), str(typ))


def _node_key(n: Dict[str, Any]) -> str:
    return str(n.get("id"))


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = int(q * (len(xs) - 1))
    return float(xs[i])


@dataclass
class SnapshotRef:
    path: str
    offset: int  # byte offset (optional)


class KGEvolutionTracker:
    def __init__(self, *, log_path: str, logger: Any):
        self.log_path = log_path
        self.logger = logger
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def snapshot_subgraph(
        self,
        *,
        query: str,
        subgraph: Dict[str, Any],
        version: str = "kg_v1",
        run_id: str = "kg_live",
        stage: str = "raw",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        nodes = subgraph.get("nodes", []) or []
        edges = (
            subgraph.get("edges", [])
            or subgraph.get("relationships", [])
            or []
        )

        confs = [
            float(e.get("confidence", 0.0))
            for e in edges
            if e.get("confidence") is not None
        ]
        has_evidence = 0
        for e in edges:
            # treat doc_hash or evidence_type as evidence-carrying
            props = e.get("properties") if isinstance(e.get("properties"), dict) else {}
            if e.get("doc_hash") or e.get("evidence_type") or props.get("doc_hash") or props.get("evidence_type"):
                has_evidence += 1
        evidence_rate = has_evidence / max(1, len(edges))

        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": "subgraph",
            "version": version,
            "run_id": run_id,
            "stage": stage,
            "scope": {
                "query": query,
                "query_hash": _hash_query(query),
            },
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "evidence_rate": float(evidence_rate),
                "confidence_p50": _quantile(confs, 0.50),
                "confidence_p10": _quantile(confs, 0.10),
                "confidence_p90": _quantile(confs, 0.90),
            },
        }
        if extra:
            rec["extra"] = extra

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def compare_snapshots(
        self, prev: Dict[str, Any], curr: Dict[str, Any]
    ) -> Dict[str, Any]:
        prev_nodes = {_node_key(n) for n in (prev.get("nodes") or [])}
        curr_nodes = {_node_key(n) for n in (curr.get("nodes") or [])}

        prev_edges = {_edge_key(e) for e in (prev.get("edges") or [])}
        curr_edges = {_edge_key(e) for e in (curr.get("edges") or [])}

        node_j = len(prev_nodes & curr_nodes) / max(
            1, len(prev_nodes | curr_nodes)
        )
        edge_j = len(prev_edges & curr_edges) / max(
            1, len(prev_edges | curr_edges)
        )

        added_edges = list(curr_edges - prev_edges)
        removed_edges = list(prev_edges - curr_edges)

        return {
            "node_jaccard": float(node_j),
            "edge_jaccard": float(edge_j),
            "added_edge_count": len(added_edges),
            "removed_edge_count": len(removed_edges),
            "added_edges_sample": [
                f"{a} -[{t}]-> {b}" for (a, b, t) in added_edges[:10]
            ],
            "removed_edges_sample": [
                f"{a} -[{t}]-> {b}" for (a, b, t) in removed_edges[:10]
            ],
            "prev_stats": prev.get("stats", {}),
            "curr_stats": curr.get("stats", {}),
        }

    def load_snapshots_for_query(
        self, *, query_hash: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_path):
            return []
        out = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if (rec.get("scope") or {}).get("query_hash") == query_hash:
                    out.append(rec)
        out.sort(key=lambda r: r.get("ts", ""))
        return out[-limit:]
