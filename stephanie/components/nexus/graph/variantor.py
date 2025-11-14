# stephanie/components/nexus/graph_variantor.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from stephanie.memory.nexus_store import NexusStore
from stephanie.components.nexus.graph.engine import NexusGraphEngine, Goal

PollinateFn = Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]
# returns a list of new scorables: [{"id": "...", "text": "...", "domains": [...], "entities": [...], "meta": {...}}, ...]

@dataclass
class VariantSpec:
    seed: int = 42
    add_pollinator_nodes: int = 0
    drop_nodes_frac: float = 0.0
    rewire_frac: float = 0.0
    perturb_sigma: float = 0.0
    graft_subgraph_size: int = 0
    temporal_jitter_frac: float = 0.0
    notes: str = ""

@dataclass
class VariantResult:
    run_id: str
    delta_V: float
    components: Dict[str, float]
    ops_applied: Dict[str, Any]
    neighbors_preview: List[Dict[str, Any]]

class GraphVariantor:
    def __init__(
        self,
        store: NexusStore,
        engine: NexusGraphEngine,
        pollinate: Optional[PollinateFn] = None,
        logger=None
    ):
        self.store = store
        self.engine = engine
        self.pollinate = pollinate
        self.log = logger

    def create_variant(self, base_run: str, variant_run: str, goal: Goal, spec: VariantSpec) -> VariantResult:
        rnd = random.Random(spec.seed)

        # 0) Start variant by copying current edges (so you mutate from baseline)
        base_edges = self.store.list_edges(base_run, limit=200000)
        _ = self.store.purge_edges(variant_run)
        self.store.write_edges(variant_run, [e.to_dict() for e in base_edges])

        ops_applied: Dict[str, Any] = {}

        # 1) ADD_POLLINATOR_NODES
        new_ids: List[str] = []
        if spec.add_pollinator_nodes and self.pollinate:
            seeds = {"goal_id": goal.id, "run_id": base_run}
            fresh = self.pollinate(goal.text, seeds)[: int(spec.add_pollinator_nodes)]
            if fresh:
                new_ids = self.engine.upsert_nodes(fresh)
                for nid in new_ids:
                    vec = self.store.get_embedding_vec(nid)
                    for nb, sim in self.store.knn(vec, k=self.engine.k_knn, min_sim=self.engine.min_sim):
                        self.store.write_edges(variant_run, [{"src": nid, "dst": nb, "type": "knn_global", "weight": float(sim)}])
            ops_applied["add_pollinator_nodes"] = len(new_ids)

        # 2) DROP_NODES
        if spec.drop_nodes_frac > 0.0:
            # drop by removing all edges touching a sampled set of nodes (scorables remain in DB)
            edges = self.store.list_edges(variant_run, limit=200000)
            nodes = list({e.src for e in edges} | {e.dst for e in edges})
            k = max(1, int(spec.drop_nodes_frac * len(nodes)))
            to_drop = set(rnd.sample(nodes, k))
            drop_list = [e for e in edges if (e.src in to_drop or e.dst in to_drop)]
            if drop_list:
                # re-write edges without dropped ones
                keep = [e for e in edges if e not in drop_list]
                self.store.purge_edges(variant_run)
                self.store.write_edges(variant_run, [e.to_dict() for e in keep])
            ops_applied["drop_nodes"] = k

        # 3) REWIRE
        if spec.rewire_frac > 0.0:
            edges = self.store.list_edges(variant_run, limit=200000)
            knn_edges = [e for e in edges if e.type == "knn_global"]
            m = max(1, int(spec.rewire_frac * len(knn_edges)))
            chosen = rnd.sample(knn_edges, m)
            # pick alternate neighbors with similar sim but from a different local cluster
            ids = list({e.src for e in chosen})
            id2vec = {i: self.store.get_embedding_vec(i) for i in ids}
            for e in chosen:
                qv = id2vec.get(e.src) or self.store.get_embedding_vec(e.src)
                alts = [(nid, sim) for nid, sim in self.store.knn(qv, k=self.engine.k_knn + 5, min_sim=self.engine.min_sim) if nid not in {e.dst, e.src}]
                if alts:
                    nid, sim = alts[rnd.randrange(0, len(alts))]
                    e.dst = nid
                    e.weight = float(sim)
            self.store.purge_edges(variant_run)
            self.store.write_edges(variant_run, [e.to_dict() for e in edges])
            ops_applied["rewire"] = m

        # 4) PERTURB_WEIGHTS
        if spec.perturb_sigma > 0.0:
            edges = self.store.list_edges(variant_run, limit=200000)
            for e in edges:
                if e.type == "knn_global":
                    noise = rnd.gauss(0.0, spec.perturb_sigma)
                    e.weight = float(min(1.0, max(0.0, e.weight + noise)))
            self.store.purge_edges(variant_run)
            self.store.write_edges(variant_run, [e.to_dict() for e in edges])
            ops_applied["perturb_sigma"] = spec.perturb_sigma

        # 5) TEMPORAL_JITTER
        if spec.temporal_jitter_frac > 0.0:
            edges = self.store.list_edges(variant_run, limit=200000)
            temp = [e for e in edges if e.type == "temporal_next"]
            m = max(1, int(spec.temporal_jitter_frac * len(temp)))
            for e in rnd.sample(temp, m):
                # swap dst within same chat neighborhood if available (cheap proxy)
                e.src, e.dst = e.dst, e.src
            self.store.purge_edges(variant_run)
            self.store.write_edges(variant_run, [e.to_dict() for e in edges])
            ops_applied["temporal_jitter"] = m

        # Evaluate value and ΔV vs baseline
        # Engine uses its own run_id; override it for this measurement
        base_engine_run = self.engine.run_id
        try:
            self.engine.run_id = base_run
            V0 = self.engine.value(goal)
            self.engine.run_id = variant_run
            V1 = self.engine.value(goal)
        finally:
            self.engine.run_id = base_engine_run

        delta = {
            "solve": V1.solve - V0.solve,
            "schema_align": V1.schema_align - V0.schema_align,
            "retention": V1.retention - V0.retention,
            "redundancy": V1.redundancy - V0.redundancy,
            "volume": V1.volume - V0.volume,
        }

        # Record a pulse on the variant head (optional)
        self.store.record_pulse(
            scorable_id=new_ids[0] if new_ids else f"{variant_run}:head",
            goal_id=goal.id,
            score=float(V1.total - V0.total),
            neighbors=[],
            subgraph_size=0,
            meta={"components": delta, "variant_run": variant_run, "spec": vars(spec), "ops": ops_applied},
        )

        return VariantResult(
            run_id=variant_run,
            delta_V=float(V1.total - V0.total),
            components=delta,
            ops_applied=ops_applied,
            neighbors_preview=[],
        )
