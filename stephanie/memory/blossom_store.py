# stephanie/memory/blossom_store.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import joinedload

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.blossom import BlossomEdgeORM, BlossomNodeORM, BlossomORM, BlossomOutputORM



class BlossomStore(BaseSQLAlchemyStore):
    """
    Store managing Blossom graphs: blossoms, nodes, edges, and common queries.
    """
    orm_model = BlossomORM
    default_order_by = "id"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "blossoms"

    # ---------- Blossom lifecycle ----------

    def create_blossom(self, blossom_dict: Dict[str, Any]) -> BlossomORM:
        def op(s):
            obj = BlossomORM(**blossom_dict)
            s.add(obj)
            s.flush()
            if self.logger:
                self.logger.log("BlossomCreated", obj.to_dict())
            return obj
        return self._run(op)

    def get(self, blossom_id: int, load_children: bool = False) -> Optional[BlossomORM]:
        def op(s):
            q = s.query(BlossomORM)
            if load_children:
                q = q.options(
                    joinedload(BlossomORM.nodes),
                    joinedload(BlossomORM.edges),
                )
            return q.filter(BlossomORM.id == blossom_id).first()
        return self._run(op)

    def list_recent(self, limit: int = 50) -> List[BlossomORM]:
        def op(s):
            return (
                s.query(BlossomORM)
                .order_by(BlossomORM.id.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def update_status(self, blossom_id: int, status: str, stats: Optional[Dict[str, Any]] = None) -> Optional[BlossomORM]:
        def op(s):
            b = s.query(BlossomORM).filter_by(id=blossom_id).first()
            if not b:
                return None
            b.status = status
            if stats is not None:
                b.stats = (b.stats or {}) | stats
            if status in ("completed", "failed", "aborted") and not b.completed_at:
                from datetime import datetime, timezone
                b.completed_at = datetime.now(timezone.utc)
            if self.logger:
                self.logger.log("BlossomStatusUpdated", {"id": blossom_id, "status": status, "stats": b.stats})
            return b
        return self._run(op)

    def delete(self, blossom_id: int) -> None:
        def op(s):
            obj = s.query(BlossomORM).filter_by(id=blossom_id).first()
            if obj:
                s.delete(obj)
                if self.logger:
                    self.logger.log("BlossomDeleted", {"id": blossom_id})
        return self._run(op)

    # ---------- Nodes ----------

    def add_node(self, node_dict: Dict[str, Any]) -> BlossomNodeORM:
        """
        node_dict requires: blossom_id
        optional: parent_id, depth, order_index, prompt_id, prompt_program_id,
                  state_text, sharpened_text, accepted, scores, features, tags, ...
        """
        def op(s):
            n = BlossomNodeORM(**node_dict)
            s.add(n)
            s.flush()
            if self.logger:
                self.logger.log("BlossomNodeInserted", n.to_dict())
            return n
        return self._run(op)

    def bulk_add_nodes(self, nodes: List[Dict[str, Any]]) -> List[BlossomNodeORM]:
        def op(s):
            objs = [BlossomNodeORM(**d) for d in nodes]
            s.add_all(objs)
            s.flush()
            if self.logger:
                for n in objs:
                    self.logger.log("BlossomNodeInserted", n.to_dict())
            return objs
        return self._run(op)

    def update_node(self, node_id: int, updates: Dict[str, Any]) -> Optional[BlossomNodeORM]:
        def op(s):
            n = s.query(BlossomNodeORM).filter_by(id=node_id).first()
            if not n:
                return None
            for k, v in updates.items():
                setattr(n, k, v)
            if self.logger:
                self.logger.log("BlossomNodeUpdated", {"id": node_id, **updates})
            return n
        return self._run(op)

    def get_node(self, node_id: int) -> Optional[BlossomNodeORM]:
        def op(s):
            return s.query(BlossomNodeORM).filter_by(id=node_id).first()
        return self._run(op)

    def get_children(self, blossom_id: int, parent_id: Optional[int]) -> List[BlossomNodeORM]:
        def op(s):
            q = s.query(BlossomNodeORM).filter_by(blossom_id=blossom_id)
            if parent_id is None:
                q = q.filter(BlossomNodeORM.parent_id.is_(None))
            else:
                q = q.filter(BlossomNodeORM.parent_id == parent_id)
            return q.order_by(BlossomNodeORM.order_index, BlossomNodeORM.id).all()
        return self._run(op)

    def get_depth(self, blossom_id: int, depth: int) -> List[BlossomNodeORM]:
        def op(s):
            return (
                s.query(BlossomNodeORM)
                .filter_by(blossom_id=blossom_id, depth=depth)
                .order_by(BlossomNodeORM.order_index, BlossomNodeORM.id)
                .all()
            )
        return self._run(op)

    # ---------- Edges ----------

    def add_edge(self, edge_dict: Dict[str, Any]) -> BlossomEdgeORM:
        """
        edge_dict requires: blossom_id, src_node_id, dst_node_id
        optional: relation, score, rationale, extra_data
        """
        def op(s):
            e = BlossomEdgeORM(**edge_dict)
            s.add(e)
            s.flush()
            if self.logger:
                self.logger.log("BlossomEdgeInserted", e.to_dict())
            return e
        return self._run(op)

    def bulk_add_edges(self, edges: List[Dict[str, Any]]) -> List[BlossomEdgeORM]:
        def op(s):
            objs = [BlossomEdgeORM(**d) for d in edges]
            s.add_all(objs)
            s.flush()
            if self.logger:
                for e in objs:
                    self.logger.log("BlossomEdgeInserted", e.to_dict())
            return objs
        return self._run(op)

    def get_edges(self, blossom_id: int) -> List[BlossomEdgeORM]:
        def op(s):
            return s.query(BlossomEdgeORM).filter_by(blossom_id=blossom_id).all()
        return self._run(op)

    # ---------- Convenience queries ----------

    def set_root(self, blossom_id: int, node_id: int) -> Optional[BlossomORM]:
        def op(s):
            b = s.query(BlossomORM).filter_by(id=blossom_id).first()
            if not b:
                return None
            b.root_node_id = node_id
            if self.logger:
                self.logger.log("BlossomRootSet", {"id": blossom_id, "root_node_id": node_id})
            return b
        return self._run(op)

    def assemble_tree(self, blossom_id: int) -> Dict[str, Any]:
        """
        Returns a compact tree dict: {root: node_dict, children: [...]}
        If multiple roots exist (no parent), returns a forest: {"forest":[...]}.
        """
        def op(s):
            nodes = (
                s.query(BlossomNodeORM)
                .filter_by(blossom_id=blossom_id)
                .order_by(BlossomNodeORM.depth, BlossomNodeORM.order_index, BlossomNodeORM.id)
                .all()
            )
            edges = s.query(BlossomEdgeORM).filter_by(blossom_id=blossom_id).all()

            by_id = {n.id: n.to_dict() for n in nodes}
            children_map: Dict[int, List[int]] = {}
            for n in nodes:
                if n.parent_id is not None:
                    children_map.setdefault(n.parent_id, []).append(n.id)

            def build(node_id: int) -> Dict[str, Any]:
                node = by_id[node_id]
                kids = [build(cid) for cid in children_map.get(node_id, [])]
                node["children"] = kids
                return node

            roots = [n.id for n in nodes if n.parent_id is None]
            if len(roots) == 1:
                return {"root": build(roots[0]), "edges": [e.to_dict() for e in edges]}
            else:
                forest = [build(r) for r in roots]
                return {"forest": forest, "edges": [e.to_dict() for e in edges]}

        return self._run(op)


    def open_episode(
        self,
        *,
        goal_text: str = "",
        seed_meta: Dict[str, Any] | None = None,
        pipeline_run_id: int | None = None,
        agent_name: str = "NexusInlineAgent",
        strategy: str = "got",
        params: Dict[str, Any] | None = None,
    ) -> BlossomORM:
        """
        Create a Blossom episode. We don't have a dedicated goal_text column,
        so we keep it in extra_data for provenance.
        """
        seed_meta = seed_meta or {}
        params = params or {}
        payload = {
            "goal_id": seed_meta.get("goal_id"),
            "pipeline_run_id": pipeline_run_id,
            "agent_name": agent_name,
            "strategy": strategy,
            "seed_type": seed_meta.get("seed_type"),
            "seed_id": seed_meta.get("seed_id"),
            "status": "running",
            "params": params,
            "extra_data": {"goal_text": goal_text, "seed_meta": seed_meta},
        }
        return self.create_blossom(payload)

    def close_episode(
        self,
        blossom_id: int,
        *,
        status: str = "completed",
        stats: Dict[str, Any] | None = None,
    ) -> Optional[BlossomORM]:
        """Mark a Blossom as completed/failed and merge stats."""
        return self.update_status(blossom_id, status=status, stats=stats or {})

    # ---------- Convenience adders (kwargs OR dict) ----------

    def add_node_kw(self, **node_kwargs) -> BlossomNodeORM:
        """
        Kwargs variant matching runner expectations.
        Maps plan_text -> state_text if provided.
        """
        # Map plan_text (runner) -> state_text (schema)
        if "plan_text" in node_kwargs and "state_text" not in node_kwargs:
            node_kwargs["state_text"] = node_kwargs.pop("plan_text")
        return self.add_node(node_kwargs)

    def add_edge_kw(self, **edge_kwargs) -> BlossomEdgeORM:
        """
        Kwargs variant matching runner expectations.
        Accepts relation='expand/refine/...' and maps keys to ORM.
        Requires blossom_id, src_node_id, dst_node_id.
        """
        return self.add_edge(edge_kwargs)

    # ---------- Bulk helpers from adapter output ----------

    def persist_forest(
        self,
        blossom_id: int,
        nodes: List[Dict[str, Any]],
        *,
        default_relation: str = "expand",
    ) -> Dict[str, int]:
        """
        Persist nodes+edges from adapter's compact node records.
        Expects records with: id, parent_id, depth, type, metric, (optional order_index).
        Returns a dict mapping external node ids -> DB ids, for callers that need it.
        """
        extid_to_dbid: Dict[str, int] = {}

        def op(s):
            # 1) Insert all nodes
            inserts = []
            for rec in nodes:
                inserts.append(
                    BlossomNodeORM(
                        blossom_id=blossom_id,
                        parent_id=None,  # set after we have db ids? we can do a second pass
                        depth=int(rec.get("depth") or 0),
                        order_index=int(rec.get("sibling_index") or rec.get("order_index") or 0),
                        state_text=rec.get("plan"),   # if adapter exposes it; ok if None
                        scores={"metric": rec.get("metric")} if rec.get("metric") is not None else None,
                        tags=[str(rec.get("type"))] if rec.get("type") else None,
                    )
                )
            s.add_all(inserts)
            s.flush()
            # Build ext->db id mapping in insertion order
            for rec, obj in zip(nodes, inserts):
                extid_to_dbid[rec["id"]] = obj.id

            # 2) Insert edges now that we have db ids
            edge_inserts = []
            for rec in nodes:
                pid_ext = rec.get("parent_id")
                if not pid_ext:
                    continue
                src = extid_to_dbid.get(pid_ext)
                dst = extid_to_dbid.get(rec["id"])
                if not src or not dst:
                    continue
                edge_inserts.append(
                    BlossomEdgeORM(
                        blossom_id=blossom_id,
                        src_node_id=src,
                        dst_node_id=dst,
                        relation=str(rec.get("type") or default_relation),
                        score=rec.get("metric"),
                    )
                )
            s.add_all(edge_inserts)
            s.flush()
            return True

        self._run(op)
        return extid_to_dbid

    # ---------- Winners & linking (compat with runner/improver) ----------

    def add_winner(
        self,
        blossom_id: int,
        path_node_ids: List[str | int],
        reward: float,
        *,
        sharpened: Dict[str, Any] | None = None,
    ) -> None:
        """
        Mark a path as a 'winner' in stats, optionally attach sharpen info to the leaf node.
        path_node_ids may be external ids (strings) or DB ids (ints).
        We'll resolve to DB ids either way.
        """
        def op(s):
            # Load blossom
            b = s.query(BlossomORM).filter_by(id=blossom_id).first()
            if not b:
                return

            # Resolve each node id to DB id if needed
            db_ids: List[int] = []
            for nid in path_node_ids:
                if isinstance(nid, int):
                    db_ids.append(nid)
                    continue
                # external -> db id resolution (best effort)
                n = (
                    s.query(BlossomNodeORM)
                    .filter_by(blossom_id=blossom_id)
                    .filter(BlossomNodeORM.extra_data["ext_id"].astext == str(nid))  # if you store ext ids; safe if absent
                    .first()
                )
                if n:
                    db_ids.append(n.id)

            # mark the leaf (last in path)
            leaf_id = db_ids[-1] if db_ids else None
            if leaf_id:
                leaf = s.query(BlossomNodeORM).filter_by(id=leaf_id).first()
                if leaf:
                    leaf.accepted = True
                    if sharpened:
                        leaf.sharpened_text = sharpened.get("sharpened")
                        leaf.sharpen_passes = int(leaf.sharpen_passes or 0) + 1
                        leaf.sharpen_gain = sharpened.get("score")
                        leaf.sharpen_meta = (leaf.sharpen_meta or {}) | {"original": sharpened.get("original")}

            # append stats
            stats = b.stats or {}
            winners = stats.get("winners", [])
            winners.append({"path": db_ids or path_node_ids, "reward": float(reward)})
            stats["winners"] = winners
            b.stats = stats
            return b

        self._run(op)

    def link_parent_children(
        self,
        *,
        parent_id: int,
        child_ids: List[int],
        relation: str = "select",
        score: float | None = None,
        rationale: str | None = None,
    ) -> None:
        """
        Create edges from parent -> each child. We infer blossom_id from the parent.
        """
        def op(s):
            parent = s.query(BlossomNodeORM).filter_by(id=parent_id).first()
            if not parent:
                return
            edges = []
            for cid in child_ids:
                edges.append(
                    BlossomEdgeORM(
                        blossom_id=parent.blossom_id,
                        src_node_id=parent_id,
                        dst_node_id=cid,
                        relation=relation,
                        score=score,
                        rationale=rationale,
                    )
                )
            s.add_all(edges)
            s.flush()
        self._run(op)

    # ---------- Small utilities ----------

    def get_blossom_id_for_node(self, node_id: int) -> Optional[int]:
        def op(s):
            n = s.query(BlossomNodeORM).filter_by(id=node_id).first()
            return n and n.blossom_id
        return self._run(op)

    def finalize(self, blossom_id: int, stats: Dict[str, Any]) -> Optional[BlossomORM]:
        """
        Mark blossom completed and store final stats (counts, win rates, deltas).
        """
        return self.update_status(blossom_id, status="completed", stats=stats)


    
    # ---------- Outputs (BlossomOutputORM) ----------

    def upsert_output_unique(self, data: Dict[str, Any]) -> BlossomOutputORM:
        """
        Upsert a BlossomOutputORM while preventing duplicates on:
          (blossom_id, role, scorable_type, scorable_id)

        Required keys in `data`:
          - blossom_id
          - role
          - scorable_type
          - scorable_id

        Optional keys:
          - source_bn_id
          - source_type
          - source_id
          - reward
          - metrics
          - notes
          - extra_data
          - episode_id (defaults to blossom_id)
        """

        required = ["blossom_id", "role", "scorable_type", "scorable_id"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required fields for BlossomOutput upsert: {missing}")

        def op(s):
            # 1) Try to find existing row with same uniqueness key
            existing = (
                s.query(BlossomOutputORM)
                .filter(
                    BlossomOutputORM.blossom_id == data["blossom_id"],
                    BlossomOutputORM.role == data["role"],
                    BlossomOutputORM.scorable_type == data["scorable_type"],
                    BlossomOutputORM.scorable_id == data["scorable_id"],
                )
                .first()
            )

            now = datetime.now(timezone.utc)

            if existing:
                # 2) Update existing record in-place
                # Only apply known columns; ignore unknown keys in `data`
                updatable_fields = {
                    "source_bn_id",
                    "source_type",
                    "source_id",
                    "reward",
                    "metrics",
                    "notes",
                    "extra_data",
                }
                for k in updatable_fields:
                    if k in data:
                        setattr(existing, k, data[k])

                # Episode id may also be updated if explicitly provided
                if "episode_id" in data:
                    existing.episode_id = data["episode_id"]

                existing.updated_at = now
                return existing

            # 3) Create new record
            payload: Dict[str, Any] = {}

            # Required uniqueness key
            payload["blossom_id"] = data["blossom_id"]
            payload["episode_id"] = data.get("episode_id", data["blossom_id"])
            payload["role"] = data["role"]
            payload["scorable_type"] = data["scorable_type"]
            payload["scorable_id"] = data["scorable_id"]

            # Optional fields if present
            for key in (
                "source_bn_id",
                "source_type",
                "source_id",
                "reward",
                "metrics",
                "notes",
                "extra_data",
            ):
                if key in data:
                    payload[key] = data[key]

            payload.setdefault("reward", 0.0)
            payload.setdefault("metrics", None)
            payload.setdefault("notes", None)
            payload.setdefault("extra_data", None)

            payload["created_at"] = now
            payload["updated_at"] = now

            obj = BlossomOutputORM(**payload)
            s.add(obj)
            s.flush()
            if self.logger:
                self.logger.log("BlossomOutputUpserted", obj.to_dict())
            return obj

        return self._run(op)

    def add_features(self, output_id: int, features: Dict[str, Any]) -> Optional[BlossomOutputORM]:
        """
        Merge a 'features' dict into BlossomOutputORM.extra_data["features"].

        If the record doesn't exist, returns None.
        """

        def op(s):
            obj = s.query(BlossomOutputORM).filter_by(id=output_id).first()
            if not obj:
                return None

            extra = dict(obj.extra_data or {})
            current_feats = dict(extra.get("features") or {})
            current_feats.update(features)
            extra["features"] = current_feats

            obj.extra_data = extra
            obj.updated_at = datetime.now(timezone.utc)

            if self.logger:
                self.logger.log(
                    "BlossomOutputFeaturesUpdated",
                    {"id": output_id, "features_keys": list(features.keys())},
                )
            return obj

        return self._run(op)

    # Small convenience helpers you will likely want downstream:

    def list_outputs_for_blossom(self, blossom_id: int) -> List[BlossomOutputORM]:
        """Return all outputs for a given Blossom episode."""
        def op(s):
            return (
                s.query(BlossomOutputORM)
                .filter(BlossomOutputORM.blossom_id == blossom_id)
                .order_by(BlossomOutputORM.id.asc())
                .all()
            )
        return self._run(op)

    def list_outputs_for_node(self, source_bn_id: int) -> List[BlossomOutputORM]:
        """Return all outputs linked to a specific BlossomNode."""
        def op(s):
            return (
                s.query(BlossomOutputORM)
                .filter(BlossomOutputORM.source_bn_id == source_bn_id)
                .order_by(BlossomOutputORM.id.asc())
                .all()
            )
        return self._run(op) 
