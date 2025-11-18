# stephanie/components/nexus/graph/knowledge_index.py
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx


@dataclass
class KnowledgeIndex:
    density: float
    clustering: float
    giant_component_ratio: float
    efficiency: float
    connectivity: float
    node_quality: float
    edge_quality: float
    score: float


def load_graph_json(path: Path) -> nx.Graph:
    """
    Loads a graph from a JSON file with flexible schemas:
      nodes: list of {id|name|key, ...}
      edges: list of {source|from, target|to, weight?}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    G = nx.DiGraph()
    nodes = data.get("nodes") or data.get("Vertices") or []
    edges = data.get("edges") or data.get("Edges") or []

    # nodes
    for nd in nodes:
        nid = (
            nd.get("id") or nd.get("name") or nd.get("key") or nd.get("label")
        )
        if nid is None:
            continue
        G.add_node(
            str(nid),
            **{
                k: v
                for k, v in nd.items()
                if k not in ("id", "name", "key", "label")
            },
        )

    # edges
    for ed in edges:
        src = ed.get("source", ed.get("from"))
        tgt = ed.get("target", ed.get("to"))
        if src is None or tgt is None:
            continue
        G.add_edge(
            str(src),
            str(tgt),
            **{
                k: v
                for k, v in ed.items()
                if k not in ("source", "from", "target", "to")
            },
        )

    return G

def compute_knowledge_index_from_files(
    baseline_path: Path,
    improved_path: Path,
    *,
    node_quality_attr: str = "quality",
    edge_quality_attr: str = "weight",
    weights: Optional[Dict[str, float]] = None,
    extra_report: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Loads two graphs from JSON, computes indices, and returns deltas.
    """
    baseG = load_graph_json(Path(baseline_path))
    impG = load_graph_json(Path(improved_path))

    base_idx = compute_knowledge_index(
        baseG,
        node_quality_attr=node_quality_attr,
        edge_quality_attr=edge_quality_attr,
        weights=weights,
    )
    imp_idx = compute_knowledge_index(
        impG,
        node_quality_attr=node_quality_attr,
        edge_quality_attr=edge_quality_attr,
        weights=weights,
    )

    # Delta per metric
    def _asdict(x):  # supports dataclass or dict
        try:
            return asdict(x)
        except Exception:
            return dict(x)

    b = _asdict(base_idx)
    i = _asdict(imp_idx)
    keys = sorted(set(b.keys()) & set(i.keys()))
    delta = {
        k: float(i[k] - b[k])
        for k in keys
        if isinstance(b[k], (int, float)) and isinstance(i[k], (int, float))
    }

    out = {
        "version": "0.1.0",
        "weights": weights or {},
        "node_quality_attr": node_quality_attr,
        "edge_quality_attr": edge_quality_attr,
        "baseline": b,
        "improved": i,
        "delta": delta,
    }

    # Optional: merge extra runtime report content (non-destructive)
    if extra_report and Path(extra_report).exists():
        try:
            out["run_report"] = json.loads(
                Path(extra_report).read_text(encoding="utf-8")
            )
        except Exception:
            pass

    return out


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_knowledge_index(
    G: nx.Graph,
    *,
    node_quality_attr: str = "quality",  # e.g. mean scorable score, clarity, etc.
    edge_quality_attr: str = "weight",  # e.g. similarity/strength
    weights: Optional[Dict[str, float]] = None,
) -> KnowledgeIndex:
    """Compute a single 'knowledge' scalar for the current graph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n <= 1:
        return KnowledgeIndex(0, 0, 0, 0, 0, 0, 0, 0)

    density = _clip(nx.density(G))  # 2m/(n(n-1)) for simple graphs

    try:
        clustering = _clip(nx.average_clustering(G.to_undirected()))
    except Exception:
        clustering = 0.0

    # Giant component ratio
    try:
        comps = sorted(
            nx.connected_components(G.to_undirected()), key=len, reverse=True
        )
        gcr = _clip(len(comps[0]) / n if comps else 0.0)
    except Exception:
        gcr = 0.0

    # Global efficiency: average 1/d(i,j)
    try:
        efficiency = _clip(nx.global_efficiency(G.to_undirected()))
    except Exception:
        efficiency = 0.0

    # Algebraic connectivity (proxy for “robustness”)
    try:
        if nx.is_connected(G.to_undirected()):
            ac = float(
                nx.algebraic_connectivity(
                    G.to_undirected(), method="tracemin_pcg"
                )
            )
            # normalize roughly by node count (soft, monotone)
            connectivity = _clip(1.0 - math.exp(-ac))
        else:
            connectivity = 0.0
    except Exception:
        connectivity = 0.0

    # Node quality (avg in [0,1] if present)
    q_values = []
    for _, data in G.nodes(data=True):
        v = data.get(node_quality_attr, None)
        if v is not None:
            q_values.append(float(v))
    node_quality = _clip(sum(q_values) / len(q_values)) if q_values else 0.0

    # Edge quality (avg weight in [0,1] if present)
    w_values = []
    for _, _, data in G.edges(data=True):
        w = data.get(edge_quality_attr, None)
        if w is not None:
            w_values.append(float(w))
    edge_quality = _clip(sum(w_values) / len(w_values)) if w_values else 0.0

    # Weighted blend → [0,1]
    W = dict(
        density=1.0,
        clustering=1.0,
        giant_component_ratio=1.0,
        efficiency=1.0,
        connectivity=1.0,
        node_quality=1.0,
        edge_quality=1.0,
    )
    if weights:
        W.update(weights)
    total_w = sum(W.values()) or 1.0
    score = (
        W["density"] * density
        + W["clustering"] * clustering
        + W["giant_component_ratio"] * gcr
        + W["efficiency"] * efficiency
        + W["connectivity"] * connectivity
        + W["node_quality"] * node_quality
        + W["edge_quality"] * edge_quality
    ) / total_w

    return KnowledgeIndex(
        density,
        clustering,
        gcr,
        efficiency,
        connectivity,
        node_quality,
        edge_quality,
        _clip(score),
    )
