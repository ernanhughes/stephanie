# stephanie/components/nexus/graph/graph.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import numpy as np

from stephanie.memory.nexus_store import NexusStore
from stephanie.models.nexus import NexusScorableORM, NexusEdgeORM, NexusMetricsORM

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Core types                                                                  #
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class NexusNode:
    """
    In-memory view of a Nexus node: a scorable + its metric vector.

    This is the "thought" unit that Nexus operates on. It intentionally does
    not expose ORM types directly so agents can construct / consume it
    without caring about the backing database.
    """
    id: str
    text: str
    target_type: Optional[str] = None
    domains: List[Any] = field(default_factory=list)
    entities: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Optional metric vector (0–1) keyed by dimension name
    dims: Dict[str, float] = field(default_factory=dict)
    # Optional cached embedding vector; NexusStore is free to ignore this
    embedding: Optional[np.ndarray] = None

    @classmethod
    def from_orm(
        cls,
        scorable: NexusScorableORM,
        metrics: Optional[NexusMetricsORM] = None,
    ) -> "NexusNode":
        vec: Dict[str, float] = {}
        if metrics is not None and metrics.vector:
            # metrics.vector is already a {dim: value} mapping; fall back to columns/values
            vec = dict(metrics.vector)
        elif metrics is not None and metrics.columns and metrics.values:
            vec = {k: float(v) for k, v in zip(metrics.columns, metrics.values)}

        return cls(
            id=scorable.id,
            text=scorable.text or "",
            target_type=scorable.target_type,
            domains=list(scorable.domains or []),
            entities=list(scorable.entities or []),
            meta=dict(scorable.meta or {}),
            dims=vec,
            embedding=None,  # filled by NexusGraph if needed
        )


@dataclass(slots=True)
class NexusEdgeSpec:
    """
    Lightweight edge spec for NexusGraph.write_edges.
    """
    src: str
    dst: str
    type: str = "related"
    weight: float = 1.0
    channels: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphScore:
    """
    Aggregated score over a Nexus run's graph.

    Not meant to be "the one true metric" – just a convenient default that can
    be extended or replaced. The .overall() is what most dashboards and blog
    examples will use.
    """
    node_count: int
    edge_count: int
    mean_dims: Dict[str, float]

    def overall(self, *, dim_weights: Optional[Dict[str, float]] = None) -> float:
        if not self.mean_dims:
            return 0.0
        if not dim_weights:
            # Simple average over all dimensions
            return float(sum(self.mean_dims.values()) / len(self.mean_dims))
        num = 0.0
        den = 0.0
        for k, w in dim_weights.items():
            if k in self.mean_dims:
                num += float(w) * float(self.mean_dims[k])
                den += float(abs(w))
        return float(num / den) if den > 0 else 0.0


@dataclass(slots=True)
class NexusViewManifest:
    """
    Small manifest that ties together a particular Nexus run's artifacts.

    This is what the blog, dashboards, and tests should consume: one file
    that points to graphs, frames, and high-level metrics.
    """
    run_id: str
    goal_preview: str = ""
    baseline_graph_path: Optional[str] = None
    improved_graph_path: Optional[str] = None
    frames_path: Optional[str] = None
    report_path: Optional[str] = None
    baseline_score: Optional[float] = None
    improved_score: Optional[float] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        import json
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

@dataclass(slots=True)
class NexusGraphConfig:
    run_id: str
    namespace: str = "nexus"
    default_edge_weight: float = 1.0

# --------------------------------------------------------------------------- #
# NexusGraph – the reusable Nexus component                                  #
# --------------------------------------------------------------------------- #

class NexusGraph:
    """
    High-level, reusable Nexus component.

    This is the "new brain" surface that agents (Pollinator, Compass, etc.)
    should talk to instead of doing ad-hoc ORM work. It wraps NexusStore and
    exposes operations in terms of NexusNode / NexusEdgeSpec / GraphScore.

    Design goals:
      - Run-scoped: every NexusGraph is bound to a run_id
      - Minimal: no scoring or LLM logic; purely graph + metrics
      - Visualization-friendly: can export Cytoscape-style JSON and manifests
    """

    def __init__(
        self,
        cfg: NexusGraphConfig,
        memory,
        logger,
        *,
        run_id: str,
    ) -> None:
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.store: NexusStore = memory.nexus
        self.run_id = run_id

    # --------------------------------------------------------------------- Nodes

    def upsert_nodes(self, nodes: Iterable[NexusNode]) -> None:
        """
        Ensure that all given nodes exist in the Nexus store, including metrics.
        """
        for node in nodes:
            row = {
                "id": node.id,
                "chat_id": node.meta.get("chat_id"),
                "turn_index": node.meta.get("turn_index"),
                "target_type": node.target_type,
                "text": node.text,
                "domains": node.domains,
                "entities": node.entities,
                "meta": {
                    k: v
                    for k, v in node.meta.items()
                    if k not in ("chat_id", "turn_index")
                },
            }
            self.store.upsert_scorable(row)

            if node.dims:
                columns = list(node.dims.keys())
                values = [float(node.dims[c]) for c in columns]
                vector = dict(node.dims)
                self.store.upsert_metrics(node.id, columns, values, vector)

            if node.embedding is not None:
                self.store.upsert_embedding(node.id, node.embedding)

    def load_node(self, scorable_id: str, *, with_metrics: bool = True) -> Optional[NexusNode]:
        s = self.store.get_scorable(scorable_id)
        if not s:
            return None

        metrics: Optional[NexusMetricsORM] = None
        if with_metrics:
            metrics = getattr(s, "metrics", None)

        return NexusNode.from_orm(s, metrics)

    def iter_nodes_for_run(self) -> Iterable[NexusNode]:
        """
        Iterate NexusNodes referenced by edges in this run.

        This gives you a graph-scoped view instead of dumping all scorables.
        """
        edges = self.store.list_edges(self.run_id, src=None, dst=None, limit=100000)
        ids: set[str] = set()
        for e in edges or []:
            ids.add(e.src)
            ids.add(e.dst)
        for sid in ids:
            node = self.load_node(sid)
            if node:
                yield node

    # --------------------------------------------------------------------- Edges

    def write_edges(self, edges: Iterable[NexusEdgeSpec]) -> int:
        edge_dicts = [
            {
                "src": e.src,
                "dst": e.dst,
                "type": e.type,
                "weight": float(e.weight),
                "channels": dict(e.channels),
            }
            for e in edges
        ]
        if not edge_dicts:
            return 0
        return self.store.write_edges(self.run_id, edge_dicts)

    def list_edges(self) -> List[NexusEdgeORM]:
        return self.store.list_edges(self.run_id, src=None, dst=None, limit=100000)

    # --------------------------------------------------------------------- KNN / neighborhood

    def knn(
        self,
        query_vec: np.ndarray,
        *,
        k: int = 25,
        min_sim: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Thin wrapper around NexusStore.knn; returns [(scorable_id, similarity)].
        """
        return self.store.knn(query_vec, k=k, min_sim=min_sim)

    # --------------------------------------------------------------------- Aggregation / scoring

    def compute_graph_score(self) -> GraphScore:
        """
        Aggregate per-node metric vectors into a simple GraphScore.

        By default, this computes the mean of each dimension across all nodes
        that have metrics.
        """
        nodes = list(self.iter_nodes_for_run())
        if not nodes:
            return GraphScore(node_count=0, edge_count=0, mean_dims={})

        # Collect all dimension names
        dim_names: set[str] = set()
        for n in nodes:
            dim_names.update(n.dims.keys())

        if not dim_names:
            return GraphScore(
                node_count=len(nodes),
                edge_count=len(self.list_edges()),
                mean_dims={},
            )

        sums: Dict[str, float] = dict.fromkeys(dim_names, 0.0)
        counts: Dict[str, int] = dict.fromkeys(dim_names, 0)

        for n in nodes:
            for d, v in n.dims.items():
                sums[d] += float(v)
                counts[d] += 1

        mean_dims = {
            d: (sums[d] / counts[d]) if counts[d] > 0 else 0.0
            for d in dim_names
        }

        edge_count = len(self.list_edges())
        return GraphScore(
            node_count=len(nodes),
            edge_count=edge_count,
            mean_dims=mean_dims,
        )

    # ------------------ ingest nodes ------------------

    def ingest_manifest(
        self,
        manifest: Dict[str, Any],
        *,
        qualities: Optional[Dict[str, float]] = None,
        metrics: Optional[Dict[str, Dict[str, float]]] = None,
        phase: str = "baseline",
    ) -> None:
        items = manifest.get("items") or []
        q = qualities or {}
        m = metrics or {}

        for item in items:
            scorable_id = item.get("scorable_id") or item.get("item_id")
            if not scorable_id:
                continue
            sid = str(scorable_id)

            near = item.get("near_identity") or {}
            row = {
                "id": sid,
                "chat_id": item.get("chat_id"),
                "turn_index": item.get("turn_index"),
                "target_type": item.get("scorable_type", "document"),
                "text": near.get("text") or "",
                "domains": item.get("domains"),
                "entities": item.get("entities"),
                "meta": {
                    "run_id": self.cfg.run_id,
                    "phase": phase,
                    "namespace": self.cfg.namespace,
                },
            }
            try:
                self.store.upsert_scorable(row)
            except Exception as e:
                log.warning("NexusGraph.upsert_scorable failed for %s: %s", sid, e)
                continue

            vec = dict(m.get(sid) or {})
            if sid in q:
                try:
                    vec.setdefault("quality", float(q[sid]))
                except Exception:
                    pass

            if not vec:
                continue

            try:
                cols = list(vec.keys())
                vals = [float(vec[c]) for c in cols]
                self.store.upsert_metrics(
                    scorable_id=sid,
                    columns=cols,
                    values=vals,
                    vector=vec,
                )
            except Exception as e:
                log.warning("NexusGraph.upsert_metrics failed for %s: %s", sid, e)

    # ------------------ ingest edges ------------------

    def write_edges_from_graph_json(
        self,
        graph_json: Dict[str, Any],
        *,
        phase: str,
        channel: str,
    ) -> None:
        if not graph_json:
            return

        edges_in = graph_json.get("edges") or []
        if not edges_in:
            return

        edges_out: List[Dict[str, Any]] = []
        for e in edges_in:
            data = e.get("data", e) or {}
            src = data.get("source") or data.get("src")
            dst = data.get("target") or data.get("dst")
            if not src or not dst:
                continue

            etype = data.get("edge_type") or data.get("type") or channel
            weight = data.get("weight", self.cfg.default_edge_weight)
            try:
                w = float(weight)
            except Exception:
                w = self.cfg.default_edge_weight

            edges_out.append(
                {
                    "src": str(src),
                    "dst": str(dst),
                    "type": str(etype),
                    "weight": w,
                    "channels": {
                        "namespace": self.cfg.namespace,
                        "phase": phase,
                        "channel": channel,
                    },
                }
            )

        if not edges_out:
            return

        try:
            self.store.write_edges(self.cfg.run_id, edges_out)
        except Exception as e:
            log.warning(
                "NexusGraph.write_edges_from_graph_json failed for run %s: %s",
                self.cfg.run_id,
                e,
            )

    # --------------------------------------------------------------------- Visualization export

    def export_cytoscape_json(
        self,
        out_path: Path,
        *,
        include_metrics: bool = True,
    ) -> None:
        """
        Export the current run's graph as a Cytoscape-style JSON structure.

        Layout is intentionally *not* computed here; let the frontend choose
        a layout algorithm. Positions are left empty.
        """
        nodes = list(self.iter_nodes_for_run())
        edges = self.list_edges()

        cy_nodes: List[Dict[str, Any]] = []
        for n in nodes:
            data: Dict[str, Any] = {
                "id": n.id,
                "label": n.meta.get("label") or (n.text[:80] if n.text else n.id),
                "target_type": n.target_type,
                "domains": n.domains,
            }
            if include_metrics and n.dims:
                data["dims"] = dict(n.dims)

            cy_nodes.append({"data": data})

        cy_edges: List[Dict[str, Any]] = []
        for e in edges or []:
            cy_edges.append(
                {
                    "data": {
                        "source": e.src,
                        "target": e.dst,
                        "type": e.type,
                        "weight": float(e.weight or 0.0),
                    }
                }
            )

        payload = {
            "run_id": self.run_id,
            "elements": {
                "nodes": cy_nodes,
                "edges": cy_edges,
            },
        }

        import json
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # --------------------------------------------------------------------- View manifest

    def build_view_manifest(
        self,
        *,
        goal_preview: str = "",
        baseline: Optional[GraphScore] = None,
        improved: Optional[GraphScore] = None,
        baseline_graph_path: Optional[Path] = None,
        improved_graph_path: Optional[Path] = None,
        frames_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
    ) -> NexusViewManifest:
        """
        Assemble a NexusViewManifest for this run.

        Pollinator (or any other agent) should call this at the end of its run
        to make the graph and metrics easy to consume in the blog / UI layer.
        """
        baseline_score = baseline.overall() if baseline else None
        improved_score = improved.overall() if improved else None

        return NexusViewManifest(
            run_id=self.run_id,
            goal_preview=goal_preview,
            baseline_graph_path=str(baseline_graph_path) if baseline_graph_path else None,
            improved_graph_path=str(improved_graph_path) if improved_graph_path else None,
            frames_path=str(frames_path) if frames_path else None,
            report_path=str(report_path) if report_path else None,
            baseline_score=baseline_score,
            improved_score=improved_score,
            extra={},
        )
