# stephanie/memory/blossom_store.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import joinedload

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.blossom import BlossomEdgeORM, BlossomNodeORM, BlossomORM


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

    def finalize(self, blossom_id: int, stats: Dict[str, Any]) -> Optional[BlossomORM]:
        """
        Mark blossom completed and store final stats (counts, win rates, deltas).
        """
        return self.update_status(blossom_id, status="completed", stats=stats)
