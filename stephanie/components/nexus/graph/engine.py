from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np

from stephanie.memory.nexus_store import NexusStore

# ---------- Types ------------------------------------------------------------

EmbedFn = Callable[[str], np.ndarray]
SolveFn = Callable[[str, Dict[str, Any]], Dict[str, float]]  
# e.g., returns {"solve_score": 0.82, "judge_score": 0.78, ...}

SchemaBuildFn = Callable[[str], Dict[str, Any]]            # SA-ICL R(x) -> Sx
SchemaRetrieveFn = Callable[[Dict[str, Any]], Dict[str, Any]]  # pick Ŝ
SchemaActivateFn = Callable[[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]

@dataclass
class Goal:
    id: str
    text: str
    meta: Dict[str, Any] = None  # e.g., dimensions, constraints, scoring weights

@dataclass
class GraphValue:
    solve: float
    schema_align: float
    retention: float
    redundancy: float
    volume: float
    total: float
    details: Dict[str, Any]

# ---------- Engine -----------------------------------------------------------

class NexusGraphEngine:
    """
    Builds a living Nexus graph in Postgres, runs goal-conditioned agents,
    and measures ΔV for candidate node insertions.
    """
    def __init__(
        self,
        cfg: dict,
        memory: Any,
        container: Any,
        solve: SolveFn,
        schema_build: SchemaBuildFn,
        schema_retrieve: SchemaRetrieveFn,
        schema_activate: SchemaActivateFn,
        *,
        run_id: str = "live",
        k_knn: int = 12,
        min_sim: float = 0.35,
        weights: Dict[str, float] = None,
        logger=None,
    ):
        
        self.cfg =cfg
        self.memory = memory
        self.container = container
        self.store: NexusStore = self.memory.nexus
        self.embed = self.memory.embedding
        self.solve = solve
        self.schema_build = schema_build
        self.schema_retrieve = schema_retrieve
        self.schema_activate = schema_activate
        self.run_id = run_id
        self.k_knn = int(k_knn)
        self.min_sim = float(min_sim)
        self.w = {"solve": 1.0, "schema": 0.3, "retention": 0.2, "redundancy": 0.2, "volume": 0.1}
        if weights:
            self.w.update(weights)
        self.log = logger

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

    def build_activated_schema(self, goal: Goal) -> Dict[str, Any]:
        Sx = self.schema_build(goal.text)          # R(x)
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

    def value(self, goal: Goal, *, schema_pack: Optional[Dict[str, Any]] = None) -> GraphValue:
        """
        Compute V(G, g) using current DB graph. 'solve' delegates to your agent stack via `solve()`.
        """
        # 1) Solve quality from your agent stack (pass run_id so agents can read graph)
        solve_metrics = self.solve(goal.text, {"run_id": self.run_id, "goal_id": goal.id})
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
        goal: Goal,
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
