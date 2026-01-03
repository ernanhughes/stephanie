# stephanie/memory/target_store.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.target import TargetInputORM, TargetORM


class TargetStore(BaseSQLAlchemyStore):
    orm_model = TargetORM
    default_order_by = TargetORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "targets"

    def insert_target(
        self,
        *,
        pipeline_run_id: Optional[int],
        target_type: str,
        target_uri: str,
        target_format: Optional[str] = None,
        title: Optional[str] = None,
        canonical_uri: Optional[str] = None,
        status: str = "created",
        content_hash: Optional[str] = None,
        root_node_type: Optional[str] = None,
        root_node_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = meta or {}

        def op(s):
            obj = TargetORM(
                pipeline_run_id=pipeline_run_id,
                target_type=target_type,
                target_format=target_format,
                title=title,
                target_uri=target_uri,
                canonical_uri=canonical_uri,
                status=status,
                content_hash=content_hash,
                root_node_type=root_node_type,
                root_node_id=root_node_id,
                meta=meta,
            )
            s.add(obj)
            s.flush()
            return obj.id

        return self._run(op)

    def mark_status(self, *, target_id: int, status: str, meta: Optional[Dict[str, Any]] = None) -> None:
        meta = meta or {}

        def op(s):
            obj = s.query(TargetORM).filter_by(id=int(target_id)).first()
            if not obj:
                return
            obj.status = status
            if meta:
                obj.meta = {**(obj.meta or {}), **meta}
            s.flush()

        self._run(op)

    def list_for_run(
        self,
        *,
        pipeline_run_id: int,
        target_type: Optional[str] = None,
        limit: int = 200,
    ) -> List[TargetORM]:
        def op(s):
            q = s.query(TargetORM).filter_by(pipeline_run_id=int(pipeline_run_id))
            if target_type:
                q = q.filter_by(target_type=target_type)
            return q.order_by(TargetORM.created_at.desc()).limit(int(limit)).all()

        return self._run(op)

    def link(
        self,
        *,
        target_id: int,
        source_id: int,
        relation_type: str,
        weight: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = meta or {}

        def op(s):
            # idempotent-ish: try find existing
            existing = (
                s.query(TargetInputORM)
                .filter_by(
                    target_id=int(target_id),
                    source_id=int(source_id),
                    relation_type=relation_type,
                )
                .first()
            )
            if existing:
                # update optional fields
                if weight is not None:
                    existing.weight = float(weight)
                if meta:
                    existing.meta = {**(existing.meta or {}), **meta}
                s.flush()
                return existing.id

            obj = TargetInputORM(
                target_id=int(target_id),
                source_id=int(source_id),
                relation_type=relation_type,
                weight=weight,
                meta=meta,
            )
            s.add(obj)
            s.flush()
            return obj.id

        return self._run(op)

    def list_inputs(self, *, target_id: int, relation_type: Optional[str] = None) -> List[TargetInputORM]:
        def op(s):
            q = s.query(TargetInputORM).filter_by(target_id=int(target_id))
            if relation_type:
                q = q.filter_by(relation_type=relation_type)
            return q.order_by(TargetInputORM.created_at.desc()).all()
        return self._run(op)
