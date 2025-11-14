# stephanie/stores/nexus_store.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sqlalchemy import desc, text, and_, or_
from sqlalchemy.orm import Session


    # -------- Helpers: nodes/edges for a run ---------------------------------

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.nexus import (NexusEdgeORM, NexusEmbeddingORM,
                                    NexusMetricsORM, NexusPulseORM,
                                    NexusScorableORM)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class NexusStore(BaseSQLAlchemyStore):
    """
    SQLAlchemy data access for the Nexus living graph:
      - scorables, embeddings, metrics
      - edges (live + batch runs)
      - pulses (audit + UI)

    Supports pgvector KNN via raw SQL when available; otherwise falls back to Python.
    """
    orm_model = NexusScorableORM
    default_order_by = "created_ts"

    def __init__(self, session_or_maker, logger=None, *, pgvector_table_prefix: str = "nexus", embedding_dim: int = 1024):
        super().__init__(session_or_maker, logger)
        self.name = "nexus"
        self.pgvector_prefix = pgvector_table_prefix
        self.embedding_dim = int(embedding_dim)

    # ------------------------------------------------------------------ Scorables

    def upsert_scorable(self, row: Dict[str, Any]) -> NexusScorableORM:
        """
        Insert or update a scorable.
        row keys: id, chat_id, turn_index, target_type, text, domains, entities, meta
        """
        def op(s: Session):
            obj = s.get(NexusScorableORM, row["id"])
            if obj is None:
                obj = NexusScorableORM(**row)
                s.add(obj)
            else:
                for k, v in row.items():
                    setattr(obj, k, v)
            s.flush()
            return obj
        return self._run(op)

    def get_scorable(self, scorable_id: str) -> Optional[NexusScorableORM]:
        def op(s: Session):
            return s.get(NexusScorableORM, scorable_id)
        return self._run(op)

    def list_scorables(self, limit: int = 1000) -> List[NexusScorableORM]:
        def op(s: Session):
            q = s.query(NexusScorableORM).order_by(desc(NexusScorableORM.created_ts)).limit(limit)
            return q.all()
        return self._run(op)

    # ------------------------------------------------------------------ Embeddings

    def upsert_embedding(self, scorable_id: str, vec: np.ndarray, *, store_norm: bool = True) -> NexusEmbeddingORM:
        def op(s: Session):
            emb = s.get(NexusEmbeddingORM, scorable_id)
            data = vec.astype(float).tolist()
            norm = float(np.linalg.norm(vec)) if store_norm else None
            if emb is None:
                emb = NexusEmbeddingORM(scorable_id=scorable_id, embed_global=data, norm_l2=norm)
                s.add(emb)
            else:
                emb.embed_global = data
                emb.norm_l2 = norm
            s.flush()
            return emb
        return self._run(op)

    def get_embedding_vec(self, scorable_id: str) -> Optional[np.ndarray]:
        def op(s: Session):
            emb = s.get(NexusEmbeddingORM, scorable_id)
            if not emb or not emb.embed_global:
                return None
            return np.array(emb.embed_global, dtype=float)
        return self._run(op)

    # ------------------------------------------------------------------ Metrics

    def upsert_metrics(self, scorable_id: str, columns: List[str], values: List[float], vector: Dict[str, float]) -> NexusMetricsORM:
        def op(s: Session):
            m = s.get(NexusMetricsORM, scorable_id)
            if m is None:
                m = NexusMetricsORM(scorable_id=scorable_id, columns=list(columns), values=list(map(float, values)), vector=dict(vector))
                s.add(m)
            else:
                m.columns = list(columns)
                m.values = list(map(float, values))
                m.vector = dict(vector)
            s.flush()
            return m
        return self._run(op)

    # ------------------------------------------------------------------ Edges

    def write_edges(self, run_id: str, edges: Iterable[Dict[str, Any]]) -> int:
        """
        Upsert edges in bulk. Edge dict keys: src, dst, type, weight, channels
        """
        edges = list(edges)
        if not edges:
            return 0

        def op(s: Session):
            count = 0
            for e in edges:
                obj = s.get(NexusEdgeORM, {"run_id": run_id, "src": e["src"], "dst": e["dst"], "type": e["type"]})
                if obj is None:
                    obj = NexusEdgeORM(
                        run_id=run_id,
                        src=e["src"],
                        dst=e["dst"],
                        type=e["type"],
                        weight=float(e.get("weight", 0.0) or 0.0),
                        channels=e.get("channels") or {},
                    )
                    s.add(obj)
                else:
                    obj.weight = float(e.get("weight", 0.0) or 0.0)
                    obj.channels = e.get("channels") or {}
                count += 1
            return count
        return int(self._run(op) or 0)

    def list_edges(self, run_id: str, *, src: Optional[str] = None, dst: Optional[str] = None, limit: int = 10000) -> List[NexusEdgeORM]:
        def op(s: Session):
            q = s.query(NexusEdgeORM).filter(NexusEdgeORM.run_id == run_id)
            if src:
                q = q.filter(NexusEdgeORM.src == src)
            if dst:
                q = q.filter(NexusEdgeORM.dst == dst)
            return q.order_by(desc(NexusEdgeORM.created_ts)).limit(limit).all()
        return self._run(op)

    def purge_edges(self, run_id: str) -> int:
        def op(s: Session):
            q = s.query(NexusEdgeORM).filter(NexusEdgeORM.run_id == run_id)
            n = q.count()
            q.delete(synchronize_session=False)
            return n
        return int(self._run(op) or 0)

    # ------------------------------------------------------------------ Pulses

    def record_pulse(self, scorable_id: str, goal_id: Optional[str], score: float, neighbors: List[Dict[str, Any]], subgraph_size: int, meta: Dict[str, Any]) -> NexusPulseORM:
        def op(s: Session):
            p = NexusPulseORM(
                scorable_id=scorable_id,
                goal_id=goal_id,
                score=float(score),
                neighbors=neighbors or [],
                subgraph_size=int(subgraph_size or 0),
                meta=meta or {},
            )
            s.add(p)
            s.flush()
            return p
        return self._run(op)

    def list_pulses(self, *, limit: int = 500) -> List[NexusPulseORM]:
        def op(s: Session):
            return s.query(NexusPulseORM).order_by(desc(NexusPulseORM.ts)).limit(limit).all()
        return self._run(op)

    # ------------------------------------------------------------------ KNN (pgvector-first with fallback)

    def knn_pgvector(self, query_vec: np.ndarray, k: int = 25, *, min_sim: float = 0.35) -> List[Tuple[str, float]]:
        """
        If you're on Postgres + pgvector with an ivfflat index, this executes native ANN search.
        Requires you to have run DDL to create a vector column and index;
        here we read JSON list and cast at query time for portability.
        """
        qlist = query_vec.astype(float).tolist()

        def op(s: Session):
            # Using 1 - cosine_distance to get similarity in [0,1]
            # embed_global is JSON; cast to vector via text function if you created a vector column separately,
            # adjust this to: ORDER BY vector_col <-> %(q)s
            sql = text(f"""
                SELECT scorable_id,
                       1 - ( (to_vector(embed_global) <=> to_vector(:q)) ) AS sim
                FROM {self.pgvector_prefix}_embedding
                ORDER BY to_vector(embed_global) <-> to_vector(:q)
                LIMIT :k
            """)
            # NOTE: `to_vector(json)` is a placeholder for your DB function that casts JSON->vector.
            # If you already store a real vector column (recommended), switch to it:
            #   SELECT scorable_id, 1 - (vector_col <=> :q) AS sim
            #   FROM nexus_embedding
            rows = []
            try:
                rows = s.execute(sql, {"q": qlist, "k": int(k)}).fetchall()
            except Exception:
                # If your DB doesn’t have the cast function, bail to fallback below
                return None
            out: List[Tuple[str, float]] = []
            for sid, sim in rows:
                if sim is not None and sim >= float(min_sim):
                    out.append((str(sid), float(sim)))
            return out

        result = self._run(op)
        if result is None:
            # Fallback to Python cosine against JSON-stored vectors
            return self.knn_python(query_vec, k=k, min_sim=min_sim)
        return result

    def knn_python(self, query_vec: np.ndarray, k: int = 25, *, min_sim: float = 0.35) -> List[Tuple[str, float]]:
        """
        Portable cosine KNN using JSON-stored vectors. Good for SQLite/DuckDB,
        slower for large corpora—use FAISS/pgvector in prod.
        """
        q = query_vec.astype(float)
        def op(s: Session):
            rows = s.query(NexusEmbeddingORM.scorable_id, NexusEmbeddingORM.embed_global).all()
            sims: List[Tuple[str, float]] = []
            for sid, jv in rows:
                if not jv:
                    continue
                v = np.array(jv, dtype=float)
                sim = _cosine(q, v)
                if sim >= float(min_sim):
                    sims.append((sid, float(sim)))
            sims.sort(key=lambda t: t[1], reverse=True)
            return sims[: int(k)]
        return self._run(op)

    # Convenience router-level helper
    def knn(self, query_vec: np.ndarray, k: int = 25, *, min_sim: float = 0.35, prefer_pgvector: bool = True) -> List[Tuple[str, float]]:
        if prefer_pgvector:
            return self.knn_pgvector(query_vec, k=k, min_sim=min_sim)
        return self.knn_python(query_vec, k=k, min_sim=min_sim)



    def list_run_nodes(self, run_id: str, limit: int = 200000) -> List[str]:
        """Return unique node ids that appear in edges for this run."""
        def op(s: Session):
            q = s.query(NexusEdgeORM.src, NexusEdgeORM.dst).filter(NexusEdgeORM.run_id == run_id)
            nodes = set()
            for src, dst in q.limit(limit).all():
                nodes.add(str(src)); nodes.add(str(dst))
            return list(nodes)
        return self._run(op) or []

    def list_edges_typed(self, run_id: str, edge_type: Optional[str] = None, limit: int = 200000) -> List[NexusEdgeORM]:
        def op(s: Session):
            q = s.query(NexusEdgeORM).filter(NexusEdgeORM.run_id == run_id)
            if edge_type:
                q = q.filter(NexusEdgeORM.type == edge_type)
            return q.order_by(desc(NexusEdgeORM.created_ts)).limit(limit).all()
        return self._run(op) or []

    def update_edge(self, run_id: str, src: str, dst: str, type_: str, *, weight: Optional[float] = None, channels: Optional[Dict[str, Any]] = None) -> bool:
        """Update a single edge if it exists."""
        def op(s: Session):
            e = s.get(NexusEdgeORM, {"run_id": run_id, "src": src, "dst": dst, "type": type_})
            if not e:
                return False
            if weight is not None: e.weight = float(weight)
            if channels is not None: e.channels = channels
            s.flush()
            return True
        return bool(self._run(op))

    def delete_edges_for_nodes(self, run_id: str, node_ids: List[str]) -> int:
        """Delete all edges touching any of the given nodes (chunked)."""
        if not node_ids: return 0
        CHUNK = 500
        total = 0
        for i in range(0, len(node_ids), CHUNK):
            chunk = node_ids[i:i+CHUNK]
            def op(s: Session):
                q = s.query(NexusEdgeORM).filter(
                    and_(NexusEdgeORM.run_id == run_id,
                         or_(NexusEdgeORM.src.in_(chunk), NexusEdgeORM.dst.in_(chunk)))
                )
                n = q.count()
                q.delete(synchronize_session=False)
                return n
            total += int(self._run(op) or 0)
        return total

    def degree_stats(self, run_id: str) -> Dict[str, int]:
        """Approximate degree per node (undirected count from edges)."""
        edges = self.list_edges(run_id, limit=200000)
        deg = {}
        for e in edges:
            deg[e.src] = deg.get(e.src, 0) + 1
            deg[e.dst] = deg.get(e.dst, 0) + 1
        return deg
