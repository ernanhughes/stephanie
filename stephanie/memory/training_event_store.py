# stephanie/memory/training_store.py (fixed, repo-style like DocumentSectionStore)
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.training_event import TrainingEventORM
from stephanie.types.training_event import TrainingEventCreate


def _sha1(s: str) -> str:

    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class TrainingEventStore(BaseSQLAlchemyStore):
    orm_model = TrainingEventORM
    default_order_by = TrainingEventORM.created_at

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "training_events"

    # ---------- Inserts (validated) ----------

    def insert_pairwise(
        self, data: dict, dedup: bool = True
    ) -> Optional[TrainingEventORM]:
        te = TrainingEventCreate(**{**data, "kind": "pairwise"})
        if not (te.pos_text and te.neg_text):
            return None
        fp = _sha1(
            f"pw||{te.model_key}||{te.dimension}||{te.query_text}||{te.pos_text}||{te.neg_text}"
        )
        return self._insert(te, fp, dedup)

    def insert_pointwise(
        self, data: dict, dedup: bool = True
    ) -> Optional[TrainingEventORM]:
        te = TrainingEventCreate(**{**data, "kind": "pointwise"})
        if not te.cand_text:
            return None
        fp = _sha1(
            f"pt||{te.model_key}||{te.dimension}||{te.query_text}||{te.cand_text}||{int(1 if te.label else 0)}"
        )
        return self._insert(te, fp, dedup)

    def _insert(
        self, te: TrainingEventCreate, fp: str, dedup: bool
    ) -> Optional[TrainingEventORM]:
        def op(s):
            if dedup:
                probe = s.query(TrainingEventORM.id).filter_by(fp=fp).first()
                if probe:
                    return None
            row = TrainingEventORM(
                model_key=te.model_key,
                dimension=te.dimension,
                kind=te.kind,
                goal_id=te.goal_id,
                pipeline_run_id=te.pipeline_run_id,
                agent_name=te.agent_name,
                query_text=te.query_text,
                pos_text=te.pos_text,
                neg_text=te.neg_text,
                cand_text=te.cand_text,
                label=None if te.label is None else int(1 if te.label else 0),
                weight=float(te.weight),
                trust=float(te.trust),
                source=te.source,
                meta=te.meta,
                fp=fp,
                processed=False,
            )
            s.add(row)
            s.flush()
            if self.logger:
                self.logger.log(
                    "TrainingEventInserted", {"id": row.id, "kind": te.kind}
                )
            return row

        return self._run(op)

    # ---------- Consumers ----------

    def sample_pairwise(
        self,
        *,
        model_key: str,
        dimension: str,
        batch_size: int = 128,
        unprocessed_only: bool = True,
    ) -> List[TrainingEventORM]:
        def op(s):
            q = s.query(TrainingEventORM).filter_by(
                model_key=model_key, dimension=dimension, kind="pairwise"
            )
            if unprocessed_only:
                q = q.filter(TrainingEventORM.processed.is_(False))
            return (
                q.order_by(TrainingEventORM.created_at.desc())
                .limit(batch_size)
                .all()
            )

        return self._run(op)

    def sample_pointwise(
        self,
        *,
        model_key: str,
        dimension: str,
        batch_size: int = 256,
        unprocessed_only: bool = True,
        pos_ratio: float = 0.5,
    ) -> List[TrainingEventORM]:
        def op(s):
            q = s.query(TrainingEventORM).filter_by(
                model_key=model_key, dimension=dimension, kind="pointwise"
            )
            if unprocessed_only:
                q = q.filter(TrainingEventORM.processed.is_(False))
            evs = (
                q.order_by(TrainingEventORM.created_at.desc())
                .limit(batch_size * 3)
                .all()
            )
            pos = [e for e in evs if (e.label or 0) == 1]
            neg = [e for e in evs if (e.label or 0) == 0]
            from random import sample

            k_pos = min(int(batch_size * pos_ratio), len(pos))
            k_neg = min(batch_size - k_pos, len(neg))
            return sample(pos, k_pos) + sample(neg, k_neg)

        return self._run(op)

    def mark_processed(self, ids: List[int]) -> None:
        if not ids:
            return

        def op(s):
            (
                s.query(TrainingEventORM)
                .filter(TrainingEventORM.id.in_(ids))
                .update(
                    {TrainingEventORM.processed: True},
                    synchronize_session=False,
                )
            )

        return self._run(op)

    def counts(self, *, model_key: str, dimension: str) -> Dict[str, int]:
        def op(s):
            base = s.query(TrainingEventORM).filter_by(
                model_key=model_key, dimension=dimension
            )
            return {
                "total": base.count(),
                "unprocessed": base.filter(
                    TrainingEventORM.processed.is_(False)
                ).count(),
                "pairwise": base.filter_by(kind="pairwise").count(),
                "pointwise": base.filter_by(kind="pointwise").count(),
            }

        return self._run(op)
