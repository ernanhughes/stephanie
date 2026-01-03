# stephanie/memory/vpm_store.py
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.vpm import VPMORM


def _robust01_vec(v: np.ndarray, lo=1.0, hi=99.0) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    if v.size == 0:
        return v
    lo_v, hi_v = np.percentile(v, [lo, hi])
    if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v <= lo_v:
        return np.zeros_like(v, dtype=np.float32)
    x = (v - lo_v) / (hi_v - lo_v + 1e-8)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _as_tensor1x(vec: np.ndarray):
    arr = np.asarray(vec, dtype=np.float32).ravel()
    if torch is not None:
        t = torch.from_numpy(arr)
        if t.ndim == 1:
            t = t.unsqueeze(0)  # (1, D)
        return t
    return arr.reshape(1, -1)


def _row_to_vec(row) -> list[float]:
    """Try common attribute names for the stored vector."""
    for attr in ("values", "vector", "embedding", "metrics"):
        if hasattr(row, attr):
            v = getattr(row, attr)
            if v is not None:
                return list(v)
    return []

class VPMStore(BaseSQLAlchemyStore):
    orm_model = VPMORM
    default_order_by = "step"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "vpms"

    def upsert_row(
        self,
        run_id: str,
        step: int,
        metric_names: List[str],
        values: List[float],
        *,
        img_png: Optional[str] = None,
        img_gif: Optional[str] = None,
        summary_json: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> VPMORM:
        def op(s):
            stmt = (
                pg_insert(VPMORM)
                .values(
                    run_id=run_id,
                    step=int(step),
                    metric_names=list(metric_names or []),
                    values=[float(x) for x in (values or [])],
                    img_png=img_png,
                    img_gif=img_gif,
                    summary_json=summary_json,
                    extra=extra or {},
                )
                .on_conflict_do_update(
                    index_elements=["run_id", "step"],
                    set_=dict(
                        metric_names=list(metric_names or []),
                        values=[float(x) for x in (values or [])],
                        img_png=img_png,
                        img_gif=img_gif,
                        summary_json=summary_json,
                        extra=extra or {},
                    ),
                )
                .returning(VPMORM.id)
            )
            s.execute(stmt)
            row = (
                s.query(VPMORM)
                .filter_by(run_id=run_id, step=int(step))
                .first()
            )
            if self.logger:
                self.logger.log(
                    "VPMRowUpserted",
                    {
                        "run_id": run_id,
                        "step": step,
                        "cols": len(metric_names or []),
                    },
                )
            return row

        return self._run(op)

    def list_by_run(
        self, run_id: str, limit: Optional[int] = None
    ) -> List[VPMORM]:
        return self._run(
            lambda: (
                self._scope()
                .query(VPMORM)
                .filter_by(run_id=run_id)
                .order_by(VPMORM.step.asc())
                .limit(limit if limit and limit > 0 else None)
                .all()
            ),
            default=[],
        )

    def get_recent(self, run_id: str, n: int = 10) -> List[VPMORM]:
        return self._run(
            lambda: (
                self._scope()
                .query(VPMORM)
                .filter_by(run_id=run_id)
                .order_by(VPMORM.step.desc())
                .limit(max(1, int(n)))
                .all()
            ),
            default=[],
        )

    def delete_run(self, run_id: str) -> int:
        def op(s):
            q = s.query(VPMORM).filter_by(run_id=run_id)
            count = q.count()
            q.delete(synchronize_session=False)
            if self.logger:
                self.logger.log(
                    "VPMRunDeleted", {"run_id": run_id, "rows": count}
                )
            return count

        return self._run(op)

    # Handy: return last N rows as (names, matrix)
    def matrix_for_run(self, run_id: str, n: Optional[int] = None):
        rows = self.list_by_run(run_id)
        if not rows:
            return [], []
        names = rows[0].metric_names or []
        if n:
            rows = rows[-int(n) :]
        mat = [r.values or [] for r in rows]
        return names, mat

    def get_random_embedding(
        self,
        *,
        strategy: str = "recent",   # "recent" | "uniform" | "any"
        normalize: bool = True,
        dim_hint: int = 8,
    ):
        """
        Return a single VPM row as an embedding (shape (1, D)).
        Recent-biased by default; falls back to synthetic if DB empty.
        """
        def op(s):
            q = s.query(self.orm_model)

            if strategy == "uniform" or strategy == "any":
                row = q.order_by(func.random()).limit(1).first()
            else:
                recent = (
                    q.order_by(getattr(self.orm_model, "created_at").desc())
                     .limit(200)
                     .all()
                )
                row = random.choice(recent) if recent else None

            if row:
                vec = _row_to_vec(row)
                if not vec:
                    vec = np.random.rand(dim_hint).tolist()
            else:
                vec = np.random.rand(dim_hint).tolist()

            arr = np.asarray(vec, dtype=np.float32)
            if normalize:
                arr = _robust01_vec(arr)
            return _as_tensor1x(arr)

        res = self._run(op)   # ✅ no 'default=' kwarg
        # If something went wrong and _run returned None, provide a safe fallback
        return res if res is not None else _as_tensor1x(np.zeros(dim_hint, dtype=np.float32))

    async def get_external_input(
        self,
        *,
        prefer_other_run: bool = True,
        normalize: bool = True,
        dim_hint: int = 8,
    ):
        """
        Asynchronous 'external' sensory input.
        Prefers a row from a different run_id than the latest; falls back to uniform; then synthetic.
        """
        def op(s):
            latest = (
                s.query(self.orm_model)
                 .order_by(getattr(self.orm_model, "created_at").desc())
                 .limit(1)
                 .first()
            )
            latest_run = getattr(latest, "run_id", None) if latest else None

            row = None
            if prefer_other_run and latest_run is not None:
                row = (
                    s.query(self.orm_model)
                     .filter(getattr(self.orm_model, "run_id") != latest_run)
                     .order_by(func.random())
                     .limit(1)
                     .first()
                )

            if row is None:
                row = (
                    s.query(self.orm_model)
                     .order_by(func.random())
                     .limit(1)
                     .first()
                )

            if row:
                vec = _row_to_vec(row)
                if not vec:
                    vec = np.random.rand(dim_hint).tolist()
            else:
                vec = np.random.rand(dim_hint).tolist()

            arr = np.asarray(vec, dtype=np.float32)
            if normalize:
                arr = _robust01_vec(arr)
            return _as_tensor1x(arr)

        # Synchronous work inside; returning from async is fine.
        res = self._run(op)
        return res if res is not None else _as_tensor1x(np.zeros(dim_hint, dtype=np.float32))
