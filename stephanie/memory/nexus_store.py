# stephanie/stores/nexus_store.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sqlalchemy import and_, desc, or_, text
from sqlalchemy.orm import Session

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.nexus import (NexusEdgeORM, NexusEmbeddingORM,
                                    NexusMetricsORM, NexusPulseORM,
                                    NexusScorableORM)
from stephanie.utils.similarity_utils import cosine

log = logging.getLogger(__name__)


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

    def __init__(
        self,
        session_or_maker,
        logger=None,
        *,
        pgvector_table_prefix: str = "nexus",
        embedding_dim: int = 1024,
    ):
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
            q = (
                s.query(NexusScorableORM)
                .order_by(desc(NexusScorableORM.created_ts))
                .limit(limit)
            )
            return q.all()

        return self._run(op)

    # ------------------------------------------------------------------ Embeddings

    def upsert_embedding(
        self,
        scorable_id: str,
        vec: Union[np.ndarray, Sequence[float]],
        *,
        store_norm: bool = True,
    ) -> NexusEmbeddingORM:
        """
        Store or update a global embedding for a scorable.

        Accepts either a NumPy array OR any sequence of floats (list, tuple, etc.).
        """
        # Coerce to ndarray so downstream math is consistent
        arr = np.asarray(vec, dtype=float)

        def op(s: Session):
            emb = s.get(NexusEmbeddingORM, scorable_id)
            data = arr.tolist()
            norm = float(np.linalg.norm(arr)) if store_norm else None

            if emb is None:
                emb = NexusEmbeddingORM(
                    scorable_id=scorable_id,
                    embed_global=data,
                    norm_l2=norm,
                )
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

    def upsert_metrics(
        self,
        scorable_id: str,
        columns: List[str],
        values: List[float],
        vector: Dict[str, float],
    ) -> NexusMetricsORM:
        def op(s: Session):
            m = s.get(NexusMetricsORM, scorable_id)
            if m is None:
                m = NexusMetricsORM(
                    scorable_id=scorable_id,
                    columns=list(columns),
                    values=list(map(float, values)),
                    vector=dict(vector),
                )
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
                obj = s.get(
                    NexusEdgeORM,
                    {
                        "run_id": run_id,
                        "src": e["src"],
                        "dst": e["dst"],
                        "type": e["type"],
                    },
                )
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

    def list_edges(
        self,
        run_id: str,
        *,
        src: Optional[str] = None,
        dst: Optional[str] = None,
        limit: int = 10000,
    ) -> List[NexusEdgeORM]:
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

    def record_pulse(
        self,
        scorable_id: str,
        goal_id: Optional[str],
        score: float,
        neighbors: List[Dict[str, Any]],
        subgraph_size: int,
        meta: Dict[str, Any],
    ) -> NexusPulseORM:
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
            return (
                s.query(NexusPulseORM)
                .order_by(desc(NexusPulseORM.ts))
                .limit(limit)
                .all()
            )

        return self._run(op)

    # ------------------------------------------------------------------ KNN (pgvector-first with fallback)

    def knn_pgvector(
        self, query_vec: np.ndarray, k: int = 25, *, min_sim: float = 0.35
    ) -> List[Tuple[str, float]]:
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

    def knn_python(
        self, query_vec: np.ndarray, k: int = 25, *, min_sim: float = 0.35
    ) -> List[Tuple[str, float]]:
        """
        Portable cosine KNN using JSON-stored vectors. Good for SQLite/DuckDB,
        slower for large corpora—use FAISS/pgvector in prod.
        """
        q = query_vec.astype(float)

        def op(s: Session):
            rows = s.query(
                NexusEmbeddingORM.scorable_id, NexusEmbeddingORM.embed_global
            ).all()
            sims: List[Tuple[str, float]] = []
            for sid, jv in rows:
                if not jv:
                    continue
                v = np.array(jv, dtype=float)
                sim = cosine(q, v)
                if sim >= float(min_sim):
                    sims.append((sid, float(sim)))
            sims.sort(key=lambda t: t[1], reverse=True)
            return sims[: int(k)]

        return self._run(op)

    # Convenience router-level helper
    def knn(
        self,
        query_vec: np.ndarray,
        k: int = 25,
        *,
        min_sim: float = 0.35,
        prefer_pgvector: bool = True,
    ) -> List[Tuple[str, float]]:
        if prefer_pgvector:
            return self.knn_pgvector(query_vec, k=k, min_sim=min_sim)
        return self.knn_python(query_vec, k=k, min_sim=min_sim)

    def list_run_nodes(self, run_id: str, limit: int = 200000) -> List[str]:
        """Return unique node ids that appear in edges for this run."""

        def op(s: Session):
            q = s.query(NexusEdgeORM.src, NexusEdgeORM.dst).filter(
                NexusEdgeORM.run_id == run_id
            )
            nodes = set()
            for src, dst in q.limit(limit).all():
                nodes.add(str(src))
                nodes.add(str(dst))
            return list(nodes)

        return self._run(op) or []

    def list_edges_typed(
        self, run_id: str, edge_type: Optional[str] = None, limit: int = 200000
    ) -> List[NexusEdgeORM]:
        def op(s: Session):
            q = s.query(NexusEdgeORM).filter(NexusEdgeORM.run_id == run_id)
            if edge_type:
                q = q.filter(NexusEdgeORM.type == edge_type)
            return q.order_by(desc(NexusEdgeORM.created_ts)).limit(limit).all()

        return self._run(op) or []

    def update_edge(
        self,
        run_id: str,
        src: str,
        dst: str,
        type_: str,
        *,
        weight: Optional[float] = None,
        channels: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a single edge if it exists."""

        def op(s: Session):
            e = s.get(
                NexusEdgeORM,
                {"run_id": run_id, "src": src, "dst": dst, "type": type_},
            )
            if not e:
                return False
            if weight is not None:
                e.weight = float(weight)
            if channels is not None:
                e.channels = channels
            s.flush()
            return True

        return bool(self._run(op))

    def delete_edges_for_nodes(self, run_id: str, node_ids: List[str]) -> int:
        """Delete all edges touching any of the given nodes (chunked)."""
        if not node_ids:
            return 0
        CHUNK = 500
        total = 0
        for i in range(0, len(node_ids), CHUNK):
            chunk = node_ids[i : i + CHUNK]

            def op(s: Session):
                q = s.query(NexusEdgeORM).filter(
                    and_(
                        NexusEdgeORM.run_id == run_id,
                        or_(
                            NexusEdgeORM.src.in_(chunk),
                            NexusEdgeORM.dst.in_(chunk),
                        ),
                    )
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

    def create_or_update_node(
        self,
        *,
        node_id: str,
        name: str,
        node_type: str,
        payload: Dict[str, Any],
        source_ids: List[str],
    ) -> NexusScorableORM:
        """
        Convenience wrapper to store a Nexus 'node' as a NexusScorableORM.

        - node_id      -> NexusScorableORM.id
        - node_type    -> NexusScorableORM.target_type
        - name         -> NexusScorableORM.text   (human-readable label)
        - domains      -> payload.get("domains", [])
        - entities     -> payload.get("entities", [])
        - meta         -> everything else, plus node-specific fields

        This keeps the storage schema unified while letting the higher-level
        Nexus graph treat nodes as first-class objects.
        """
        # Optional fields we know how to surface
        domains = payload.get("domains") or []
        entities = payload.get("entities") or []

        # Start meta with any existing meta then enrich
        base_meta = payload.get("meta") or {}
        meta: Dict[str, Any] = dict(base_meta)

        # Carry through useful node attributes if present in payload
        # (these are the LightRAG-style bits / key-value profile)
        if "index_keys" in payload:
            meta["index_keys"] = payload["index_keys"]
        if "value_summary" in payload:
            meta["value_summary"] = payload["value_summary"]
        if "attributes" in payload:
            meta["attributes"] = payload["attributes"]

        # Always stamp core node identity into meta
        meta.update(
            {
                "node_type": node_type,
                "node_name": name,
                "source_ids": list(source_ids or []),
            }
        )

        row = {
            "id": node_id,
            "chat_id": payload.get("chat_id"),  # optional, ok if None
            "turn_index": payload.get("turn_index"),  # optional, ok if None
            "target_type": node_type,
            "text": name,
            "domains": domains,
            "entities": entities,
            "meta": meta,
        }

        return self.upsert_scorable(row)

    def get_node(self, node_id: str) -> Optional[NexusScorableORM]:
        """Alias for get_scorable, but semantically 'node'."""
        return self.get_scorable(node_id)

    def list_nodes_for_scorable(
        self,
        scorable_id: str,
        limit: int = 200,
    ) -> List[NexusScorableORM]:
        """
        Return nodes whose meta['source_ids'] contains the given scorable_id.
        This is a simple Python-side filter for now; we can optimize with
        JSONB queries later if needed.
        """

        def op(s: Session):
            q = s.query(NexusScorableORM).order_by(
                desc(NexusScorableORM.created_ts)
            )
            rows = q.limit(5000).all()  # crude but fine for now
            result: List[NexusScorableORM] = []
            for row in rows:
                meta = row.meta or {}
                src_ids = meta.get("source_ids") or []
                if scorable_id in src_ids:
                    result.append(row)
                    if len(result) >= limit:
                        break
            return result

        return self._run(op)

    def list_edges_for_node(
        self,
        run_id: str,
        node_id: str,
        *,
        limit: int = 1000,
    ) -> List[NexusEdgeORM]:
        """
        Return edges where this node appears as either src or dst.
        """

        def op(s: Session):
            q = s.query(NexusEdgeORM).filter(
                and_(
                    NexusEdgeORM.run_id == run_id,
                    or_(
                        NexusEdgeORM.src == node_id,
                        NexusEdgeORM.dst == node_id,
                    ),
                )
            )
            return q.order_by(desc(NexusEdgeORM.created_ts)).limit(limit).all()

        return self._run(op) or []

    def log_local_tree(
        self,
        run_id: str,
        root_id: str,
        *,
        depth: int = 2,
        max_per_level: int = 8,
    ) -> None:
        """
        Debug helper: log a small ASCII tree around `root_id` using stored edges.

        - Treats edges as undirected for traversal.
        - Uses BFS up to `depth` hops.
        - Caps the number of children explored per node via `max_per_level`.
        - Optionally enriches node labels with scorable target_type + text preview.

        Example:

            memory.nexus.log_local_tree(
                run_id="kg_live",
                root_id="paper:2501.12345#sec-003",
                depth=2,
                max_per_level=6,
                logger=self.logger,
            )
        """
        from collections import defaultdict, deque

        try:
            # --------- 1) BFS to collect neighborhood structure ---------
            children: Dict[str, List[Tuple[str, str, float]]] = defaultdict(
                list
            )
            seen: set[str] = set([root_id])
            frontier: deque[Tuple[str, int]] = deque()
            frontier.append((root_id, 0))

            # We’ll only call list_edges, which you already have:
            # def list_edges(self, run_id, *, src=None, dst=None, limit=10000) -> List[NexusEdgeORM]

            while frontier:
                node_id, dist = frontier.popleft()
                if dist >= depth:
                    continue

                # Fetch edges where this node is src OR dst
                edges_src = self.list_edges(
                    run_id, src=node_id, limit=max_per_level * 2
                )
                edges_dst = self.list_edges(
                    run_id, dst=node_id, limit=max_per_level * 2
                )

                neighbors: List[Tuple[str, str, float]] = []
                for e in edges_src:
                    neighbor = e.dst
                    edge_type = getattr(e, "type", "edge")
                    weight = float(getattr(e, "weight", 1.0) or 1.0)
                    neighbors.append((neighbor, edge_type, weight))

                for e in edges_dst:
                    neighbor = e.src
                    edge_type = getattr(e, "type", "edge")
                    weight = float(getattr(e, "weight", 1.0) or 1.0)
                    neighbors.append((neighbor, edge_type, weight))

                # Deduplicate neighbors for this node
                uniq: Dict[str, Tuple[str, float]] = {}
                for nid, etype, w in neighbors:
                    if nid not in uniq:
                        uniq[nid] = (etype, w)
                neighbors = [
                    (nid, etype, w) for nid, (etype, w) in uniq.items()
                ]

                # Cap per-node fan-out
                neighbors = neighbors[:max_per_level]

                # Store and enqueue unseen neighbors
                for nid, etype, w in neighbors:
                    children[node_id].append((nid, etype, w))
                    if nid not in seen:
                        seen.add(nid)
                        frontier.append((nid, dist + 1))

            # --------- 2) Enrich labels with scorable text (optional) ----

            all_ids: set[str] = set([root_id])
            for parent, kids in children.items():
                all_ids.add(parent)
                for nid, _, _ in kids:
                    all_ids.add(nid)

            labels: Dict[str, str] = {}
            for nid in all_ids:
                row = self.get_scorable(nid)
                if row is None:
                    labels[nid] = nid
                    continue

                target_type = getattr(row, "target_type", None) or ""
                text = (row.text or "").replace("\n", " ").strip()
                if len(text) > 80:
                    text = text[:77] + "..."
                if target_type:
                    labels[nid] = f"{nid} [{target_type}] {text}"
                else:
                    labels[nid] = f"{nid} {text}"

            # --------- 3) Build ASCII tree -------------------------------

            lines: List[str] = []
            header = (
                f"NexusLocalTree[{root_id}] (run_id={run_id}, depth={depth})"
            )
            lines.append(header)

            root_label = labels.get(root_id, root_id)
            lines.append(f"root: {root_label}")

            def render(node_id: str, prefix: str = "") -> None:
                kids = children.get(node_id, [])
                for i, (child_id, edge_type, weight) in enumerate(kids):
                    connector = "└─" if i == len(kids) - 1 else "├─"
                    edge_str = edge_type
                    try:
                        edge_str = f"{edge_type} (w={float(weight):.2f})"
                    except Exception:
                        pass

                    child_label = labels.get(child_id, child_id)
                    lines.append(
                        f"{prefix}{connector} {edge_str} → {child_label}"
                    )

                    # Extend prefix for grandchildren
                    next_prefix = prefix + (
                        "   " if i == len(kids) - 1 else "│  "
                    )
                    render(child_id, next_prefix)

            render(root_id, prefix="  ")

            log.info("\n%s", "\n".join(lines))

        except Exception as e:
            log.warning(
                "NexusLocalTreeError run_id=%s root_id=%s error=%s",
                run_id,
                root_id,
                str(e),
            )
