# stephanie/memory/training_store.py
from __future__ import annotations

import hashlib
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.training_event import TrainingEventORM


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class TrainingEventStore(BaseSQLAlchemyStore):
    orm_model = TrainingEventORM
    default_order_by = TrainingEventORM.created_at


    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "training_events"

    def name(self) -> str:
        return self.name

    # ---- Emitters -----------------------------------------------------------
    def add_pairwise(
        self,
        *,
        model_key: str,
        dimension: str,
        query_text: str,
        pos_text: str,
        neg_text: str,
        weight: float = 1.0,
        trust: float = 0.0,
        goal_id: Optional[str] = None,
        pipeline_run_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        source: str = "memento",
        meta: Optional[Dict] = None,
        dedup: bool = True,
    ) -> Optional[int]:
        if not (pos_text and neg_text):
            return None
        fp_base = f"pw||{model_key}||{dimension}||{query_text}||{pos_text}||{neg_text}"
        fp = _sha1(fp_base)
        if dedup and self._exists_fp(fp):
            return None
        ev = TrainingEventORM(
            model_key=model_key,
            dimension=dimension,
            kind="pairwise",
            goal_id=goal_id,
            pipeline_run_id=pipeline_run_id,
            agent_name=agent_name,
            query_text=query_text,
            pos_text=pos_text,
            neg_text=neg_text,
            weight=float(weight),
            trust=float(trust),
            source=source,
            meta=(meta or {}),
            fp=fp,
            processed=False,
        )
        self.session.add(ev)
        self.session.commit()
        return ev.id

    def add_pointwise(
        self,
        *,
        model_key: str,
        dimension: str,
        query_text: str,
        cand_text: str,
        label: int,
        weight: float = 1.0,
        trust: float = 0.0,
        goal_id: Optional[str] = None,
        pipeline_run_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        source: str = "memento",
        meta: Optional[Dict] = None,
        dedup: bool = True,
    ) -> Optional[int]:
        if not cand_text:
            return None
        label = int(1 if label else 0)
        fp_base = (
            f"pt||{model_key}||{dimension}||{query_text}||{cand_text}||{label}"
        )
        fp = _sha1(fp_base)
        if dedup and self._exists_fp(fp):
            return None
        ev = TrainingEventORM(
            model_key=model_key,
            dimension=dimension,
            kind="pointwise",
            goal_id=goal_id,
            pipeline_run_id=pipeline_run_id,
            agent_name=agent_name,
            query_text=query_text,
            cand_text=cand_text,
            label=label,
            weight=float(weight),
            trust=float(trust),
            source=source,
            meta=(meta or {}),
            fp=fp,
            processed=False,
        )
        self.session.add(ev)
        self.session.commit()
        return ev.id

    def _exists_fp(self, fp: str) -> bool:
        return (
            self.session.query(TrainingEventORM.id).filter_by(fp=fp).first()
            is not None
        )

    # ---- Consumers ----------------------------------------------------------
    def sample_pairwise(
        self,
        *,
        model_key: str,
        dimension: str,
        batch_size: int = 128,
        unprocessed_only: bool = True,
        pos_ratio: float = 0.5,  # unused here, but you might use for pointwise
    ) -> List[TrainingEventORM]:
        q = self.session.query(TrainingEventORM).filter_by(
            model_key=model_key, dimension=dimension, kind="pairwise"
        )
        if unprocessed_only:
            q = q.filter(TrainingEventORM.processed.is_(False))
        # For simplicity: recent-first; or ORDER BY random() for more variety
        q = q.order_by(TrainingEventORM.created_at.desc())
        return q.limit(batch_size).all()

    def sample_pointwise(
        self,
        *,
        model_key: str,
        dimension: str,
        batch_size: int = 256,
        unprocessed_only: bool = True,
        pos_ratio: float = 0.5,
    ) -> List[TrainingEventORM]:
        q = self.session.query(TrainingEventORM).filter_by(
            model_key=model_key, dimension=dimension, kind="pointwise"
        )
        if unprocessed_only:
            q = q.filter(TrainingEventORM.processed.is_(False))
        q = q.order_by(TrainingEventORM.created_at.desc()).limit(
            batch_size * 3
        )
        evs = q.all()
        # Balance pos/neg in Python
        pos = [e for e in evs if (e.label or 0) == 1]
        neg = [e for e in evs if (e.label or 0) == 0]
        k_pos = min(int(batch_size * pos_ratio), len(pos))
        k_neg = min(batch_size - k_pos, len(neg))
        return random.sample(pos, k_pos) + random.sample(neg, k_neg)

    def mark_processed(self, ids: List[int]):
        if not ids:
            return
        (
            self.session.query(TrainingEventORM)
            .filter(TrainingEventORM.id.in_(ids))
            .update(
                {TrainingEventORM.processed: True}, synchronize_session=False
            )
        )
        self.session.commit()

    # ---- Stats for controller ----------------------------------------------
    def counts(self, *, model_key: str, dimension: str) -> Dict[str, int]:
        q_all = self.session.query(TrainingEventORM).filter_by(
            model_key=model_key, dimension=dimension
        )
        q_unp = q_all.filter(TrainingEventORM.processed.is_(False))
        return {
            "total": q_all.count(),
            "unprocessed": q_unp.count(),
            "pairwise": q_all.filter_by(kind="pairwise").count(),
            "pointwise": q_all.filter_by(kind="pointwise").count(),
        }
