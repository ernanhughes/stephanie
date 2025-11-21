# stephanie/components/nexus/graph/graph.py
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from stephanie.memory.nexus_store import NexusStore
from stephanie.models.nexus import (NexusEdgeORM, NexusMetricsORM,
                                    NexusScorableORM)
from stephanie.data.scorable_row import ScorableRow
import json
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Core types                                                                  #
# --------------------------------------------------------------------------- #

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(slots=True)
class GraphValue:
    """
    Represents the multi-dimensional 'value' of a Nexus graph state with respect to a goal.

    This is the output of V(G, g) — a scalar total + breakdown by cognitive dimensions.
    Designed for:
      - Comparing graph states (before/after edits)
      - Driving agent decisions (e.g., which Scorable to expand next)
      - Visualizing progress in blogs/UIs ("filmstrip" of value evolution)

    The total score is a weighted combination; components allow introspection.
    """
    solve: float = 0.0          # Quality of direct answer generation
    schema_align: float = 0.0   # Alignment with expected reasoning/schema structure
    retention: float = 0.0      # Connectedness / integration into existing knowledge
    redundancy: float = 0.0     # Overlap/duplication penalty (negative contributor)
    volume: float = 0.0         # Cognitive reach / depth of exploration

    total: float = 0.0          # Final aggregated score: V(G, g)

    details: Optional[Dict[str, Any]] = None  # Optional debug/metadata (e.g., episodic hits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solve": self.solve,
            "schema_align": self.schema_align,
            "retention": self.retention,
            "redundancy": self.redundancy,
            "volume": self.volume,
            "total": self.total,
            "details": self.details or {},
        }

    def __add__(self, other: "GraphValue") -> "GraphValue":
        return GraphValue(
            solve=self.solve + other.solve,
            schema_align=self.schema_align + other.schema_align,
            retention=self.retention + other.retention,
            redundancy=self.redundancy + other.redundancy,
            volume=self.volume + other.volume,
            total=self.total + other.total,
            details={
                "components": {
                    k: [getattr(self, k), getattr(other, k)]
                    for k in ("solve", "schema_align", "retention", "redundancy", "volume")
                }
            },
        )

    def __mul__(self, scalar: float) -> "GraphValue":
        return GraphValue(
            solve=self.solve * scalar,
            schema_align=self.schema_align * scalar,
            retention=self.retention * scalar,
            redundancy=self.redundancy * scalar,
            volume=self.volume * scalar,
            total=self.total * scalar,
            details=self.details,
        )
    
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
    ) -> NexusNode:
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
        container,
        logger,
        *,
        run_id: str,
    ) -> None:
        self.cfg = cfg
        self.memory = memory
        self.container = container
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
        val = graph_json
        if isinstance(val, str):
            val = json.loads(Path(val).read_text())

        edges_in = val.get("edges") or []
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

    # ---- Graph construction -------------------------------------------------

    def upsert_nodes(self, scorables: Iterable[Dict[str, Any]]) -> List[str]:
        """
        Upsert scorables + embeddings + basic metrics into the DB. Returns list of ids.
        Required scorable keys: id, text (+ optional domains/entities/meta/turn_index/chat_id)
        """
        ids = [] 
        for row in scorables:
            s = self.store.upsert_scorable(row)
            vec = self.embed(row.get("text") or "")
            self.store.upsert_embedding(s.id, vec)
            ids.append(s.id)
        return ids

    def rebuild_edges(self, *, add_knn=True, add_temporal=True, channel_name="knn_global") -> int:
        """
        Recompute edges for all nodes in this run snapshot.
        - KNN edges from global embeddings
        - Temporal edges by chat_id/turn_index (if present)
        """
        scorables = self.store.list_scorables(limit=100000)
        id2vec: Dict[str, np.ndarray] = {}
        for s in scorables:
            v = self.store.get_embedding_vec(s.id)
            if v is not None:
                id2vec[s.id] = v

        edges: List[Dict[str, Any]] = []
        if add_knn:
            for sid, qv in id2vec.items():
                nbrs = self.store.knn(qv, k=self.k_knn, min_sim=self.min_sim)
                for nid, sim in nbrs:
                    if nid == sid: 
                        continue
                    edges.append({"src": sid, "dst": nid, "type": channel_name, "weight": float(sim), "channels": {"sim": sim}})

        if add_temporal:
            # add (chat_id, turn_index) → temporal_next
            by_chat: Dict[str, List[Any]] = {}
            for s in scorables:
                if s.chat_id is not None and s.turn_index is not None:
                    by_chat.setdefault(s.chat_id, []).append(s)
            for chat_id, rows in by_chat.items():
                rows.sort(key=lambda r: int(r.turn_index))
                for a, b in zip(rows, rows[1:]):
                    edges.append({"src": a.id, "dst": b.id, "type": "temporal_next", "weight": 1.0})

        self.store.purge_edges(self.run_id)
        return self.store.write_edges(self.run_id, edges)

    # ---- SA-ICL schema pipeline --------------------------------------------

    def build_activated_schema(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        Sx = self.schema_build(goal.get("goal_text"))          # R(x)
        S_hat = self.schema_retrieve(Sx)           # Ŝ
        episodic = self._select_episodic(S_hat)    # Ê_τ (decay-aware)
        S_new = self.schema_activate(Sx, S_hat, episodic)  # f(Sx, Ŝ, Ê_τ)
        return {"Sx": Sx, "S_hat": S_hat, "episodic": episodic, "S_new": S_new}

    def _select_episodic(self, S_hat: Dict[str, Any], tau: float = 0.6) -> List[Dict[str, Any]]:
        """
        Minimal stub: pull recent/topical scorables and filter by (schema, time) weights.
        You can later wire your MemCube/decay function here.
        """
        out: List[Dict[str, Any]] = []
        for s in self.store.list_scorables(limit=512):
            # placeholder association weight using cosine on embeddings as a proxy
            v = self.store.get_embedding_vec(s.id)
            if v is None or "schema_vec" not in S_hat:
                continue
            sim = float(np.dot(v, S_hat["schema_vec"]) / (np.linalg.norm(v) * (np.linalg.norm(S_hat["schema_vec"]) + 1e-9)))
            if sim >= tau:
                out.append({"id": s.id, "text": s.text, "assoc": sim})
        return out

    # ---- Value function -----------------------------------------------------

    def value(self, goal: Dict[str, Any], *, schema_pack: Optional[Dict[str, Any]] = None) -> GraphValue:
        """
        Compute V(G, g) using current DB graph. 'solve' delegates to your agent stack via `solve()`.
        """
        # 1) Solve quality from your agent stack (pass run_id so agents can read graph)
        solve_metrics = self.solve(goal.get("goal_text"), {"run_id": self.run_id, "goal_id": goal.get("goal_id")})
        solve_score = float(solve_metrics.get("solve_score", 0.0))

        # 2) Schema alignment (e.g., cosine or KL vs an expected schema template)
        schema_align = 0.0
        if not schema_pack:
            schema_pack = self.build_activated_schema(goal)
        # Simple proxy: how concentrated episodic set is, and how close Sx->S_new moved in right direction
        schema_align = float(min(1.0, 0.5 + 0.5 * (len(schema_pack.get("episodic", [])) / 10.0)))

        # 3+4) Retention/Redundancy from GoT doc-merge idea (use fast heuristics here; exact uses an LLM judge)
        # Heuristic: high avg edge weight among selected subgraph neighbors -> more redundancy; coverage -> retention
        retention, redundancy = self._retention_redundancy_heuristic()

        # 5) Volume proxy: reachability to top-k “answer-like” nodes (temporal sinks or high-solve nodes)
        volume = self._volume_proxy()

        total = (self.w["solve"] * solve_score
                + self.w["schema"] * schema_align
                + self.w["retention"] * retention
                - self.w["redundancy"] * redundancy
                + self.w["volume"] * volume)

        return GraphValue(
            solve=solve_score,
            schema_align=schema_align,
            retention=retention,
            redundancy=redundancy,
            volume=volume,
            total=float(total),
            details={"solve_metrics": solve_metrics, "schema": schema_pack},
        )

    def _retention_redundancy_heuristic(self) -> Tuple[float, float]:
        """
        Lightweight proxy:
        - retention ~ fraction of nodes with degree>0 inside current run (connectedness)
        - redundancy ~ mean weight among KNN edges above a high similarity threshold
        """
        edges = self.store.list_edges(self.run_id, limit=200000)
        if not edges:
            return 0.0, 0.0
        deg = {}
        sims = []
        for e in edges:
            deg[e.src] = deg.get(e.src, 0) + 1
            deg[e.dst] = deg.get(e.dst, 0) + 1
            if e.type == "knn_global":
                sims.append(float(e.weight))
        retention = min(1.0, sum(1 for d in deg.values() if d > 0) / max(1, len(deg)))
        redundancy = float(np.mean([s for s in sims if s >= 0.9])) if sims else 0.0
        return retention, redundancy

    def _volume_proxy(self, topk: int = 10) -> float:
        """
        Approximate GoT 'volume of a thought' as: average # of predecessors of the top-k nodes by degree.
        """
        edges = self.store.list_edges(self.run_id, limit=200000)
        if not edges:
            return 0.0
        indeg = {}
        pred = {}
        for e in edges:
            indeg[e.dst] = indeg.get(e.dst, 0) + 1
            pred.setdefault(e.dst, set()).add(e.src)
        top = sorted(indeg.items(), key=lambda t: t[1], reverse=True)[:topk]
        vols = [len(pred.get(n, [])) for n, _ in top]
        if not vols:
            return 0.0
        return min(1.0, float(np.mean(vols) / 25.0))  # normalize

    # ---- ΔV measurement -----------------------------------------------------

    def delta_value_for_addition(
        self,
        candidate: Dict[str, Any],
        goal: Dict[str, Any],
        *,
        shapley_samples: int = 0,
        record_pulse: bool = True
    ) -> Dict[str, Any]:
        """
        1) Measure V(G,g)
        2) Insert candidate node + vec; add edges; measure V(G', g)
        3) Δ = V' - V; optional Shapley-lite by random insertion orders
        4) Persist a NexusPulse record if requested
        """
        baseline = self.value(goal)

        # Insert node + edges (idempotent upsert)
        self.upsert_nodes([candidate])
        vec = self.store.get_embedding_vec(candidate["id"])
        # Add candidate’s outward KNN edges + temporal if turn_index present
        nbrs = self.store.knn(vec, k=self.k_knn, min_sim=self.min_sim)
        new_edges = [{"src": candidate["id"], "dst": nid, "type": "knn_global", "weight": sim} for nid, sim in nbrs]
        self.store.write_edges(self.run_id, new_edges)

        after = self.value(goal)

        delta = {
            "candidate_id": candidate["id"],
            "V_before": baseline.total,
            "V_after": after.total,
            "ΔV": after.total - baseline.total,
            "components": {
                "solve": after.solve - baseline.solve,
                "schema_align": after.schema_align - baseline.schema_align,
                "retention": after.retention - baseline.retention,
                "redundancy": after.redundancy - baseline.redundancy,
                "volume": after.volume - baseline.volume,
            },
            "baseline": baseline.details,
            "after": after.details,
        }

        # Optional: Shapley-lite (few permutations of local neighborhood)
        if shapley_samples > 0:
            delta["shapley_lite"] = self._shapley_lite(candidate, goal, samples=shapley_samples)

        if record_pulse:
            self.store.record_pulse(
                scorable_id=candidate["id"],
                goal_id=goal.id,
                score=float(delta["ΔV"]),
                neighbors=[{"nid": nid, "sim": float(sim)} for nid, sim in nbrs],
                subgraph_size=0,
                meta={"components": delta["components"], "run_id": self.run_id},
            )

        return delta

    def _shapley_lite(self, candidate: Dict[str, Any], goal: Goal, samples: int = 4) -> Dict[str, float]:
        """
        Sample a few permutations of the candidate’s KNN neighborhood and average marginal gains.
        """
        import random
        vec = self.store.get_embedding_vec(candidate["id"])
        nbrs = [nid for nid, _ in self.store.knn(vec, k=8, min_sim=self.min_sim)]
        if not nbrs:
            return {"samples": 0, "mean": 0.0}
        gains: List[float] = []
        for _ in range(samples):
            order = nbrs[:]
            random.shuffle(order)
            # Baseline without candidate
            base = self.value(goal).total
            # Simulate “adding” neighbors then candidate (cheap proxy)
            _ = order  # no-op hook if you later add staged edges
            with_cand = self.value(goal).total  # measure again (kept simple; you can cache)
            gains.append(with_cand - base)
        return {"samples": samples, "mean": float(np.mean(gains))}
