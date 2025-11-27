# stephanie/memory/metric_store.py
from __future__ import annotations

import datetime
from typing import List, Optional

from sqlalchemy import desc

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.metrics import (MetricDeltaORM, MetricGroupORM,
                                      MetricVectorORM, MetricVPMORM)


class MetricStore(BaseSQLAlchemyStore):
    """
    MetricStore provides all persistence for:
      - MetricGroupORM (pipeline runs)
      - MetricVectorORM (metrics per scorable)
      - MetricDeltaORM (diffs)
      - MetricVPMORM (images)

    Pattern identical to ChatStore.
    """

    orm_model = MetricGroupORM
    default_order_by = "created_at"

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "metrics"

    # ============================================================
    # GROUP OPERATIONS
    # ============================================================

    def create_group(self, run_id: str, meta: dict) -> MetricGroupORM:
        def op(s):
            g = MetricGroupORM(run_id=run_id, meta=meta or {})
            s.add(g)
            s.flush()
            return g
        return self._run(op)

    def get_group(self, run_id: str) -> Optional[MetricGroupORM]:
        def op(s):
            return s.query(MetricGroupORM).filter_by(run_id=run_id).one_or_none()
        return self._run(op)




    # ---------- Existence checks by scorable ----------
    def vector_exists_by_scorable(self, scorable_id: str, scorable_type: str) -> bool:
        def op(s):
            q = (s.query(MetricVectorORM.id)
                   .filter(MetricVectorORM.scorable_id == scorable_id,
                           MetricVectorORM.scorable_type == scorable_type)
                   .limit(1))
            return s.query(q.exists()).scalar()
        return self._run(op)

    # ---------- Insert-or-update by scorable ----------
    def save_or_update_vector_by_scorable(
        self,
        scorable_id: str,
        scorable_type: str,
        metrics: dict,
        reduced: dict,
        *,
        run_id: Optional[str] = None,   # still recorded when present
    ) -> MetricVectorORM:
        """
        Idempotent: guarantees a single row per (scorable_id, scorable_type).
        Updates metrics/reduced if the row already exists.
        """
        def op(s):
            row = (
                s.query(MetricVectorORM)
                 .filter(MetricVectorORM.scorable_id == scorable_id,
                         MetricVectorORM.scorable_type == scorable_type)
                 .one_or_none()
            )
            if row is None:
                row = MetricVectorORM(
                    run_id=run_id,
                    scorable_id=scorable_id,
                    scorable_type=scorable_type,
                    metrics=metrics or {},
                    reduced=reduced or {},
                )
                s.add(row)
                s.flush()
                return row

            # update in place (keep original created_at)
            if run_id and not row.run_id:
                row.run_id = run_id
            row.metrics = metrics or {}
            row.reduced = reduced or {}
            s.flush()
            return row
        return self._run(op)

    # Optional: “skip if exists” read check for fast short-circuiting
    def should_skip_vector_for_scorable(
        self,
        scorable_id: str,
        scorable_type: str,
        *,
        skip_if_exists: bool = True
    ) -> bool:
        return skip_if_exists and self.vector_exists_by_scorable(scorable_id, scorable_type)


    # ============================================================
    # VECTOR OPERATIONS
    # ============================================================

    def save_vector(
        self,
        run_id: str,
        scorable_id: str,
        scorable_type: str,
        metrics: dict,
        reduced: dict,
    ) -> MetricVectorORM:
        def op(s):
            v = MetricVectorORM(
                run_id=run_id,
                scorable_id=scorable_id,
                scorable_type=scorable_type,
                metrics=metrics or {},
                reduced=reduced or {},
            )
            s.add(v)
            s.flush()
            return v
        return self._run(op)

    def get_vectors_for_scorable(self, run_id: str, scorable_id: str) -> List[MetricVectorORM]:
        def op(s):
            return (
                s.query(MetricVectorORM)
                .filter_by(run_id=run_id, scorable_id=scorable_id)
                .all()
            )
        return self._run(op)

    def get_vectors_for_run(self, run_id: str) -> List[MetricVectorORM]:
        def op(s):
            return s.query(MetricVectorORM).filter_by(run_id=run_id).all()
        return self._run(op)

    # ============================================================
    # DELTA OPERATIONS
    # ============================================================

    def save_delta(
        self,
        run_id: str,
        scorable_id: str,
        scorable_type: str,
        deltas: dict,
    ) -> MetricDeltaORM:
        def op(s):
            d = MetricDeltaORM(
                run_id=run_id,
                scorable_id=scorable_id,
                scorable_type=scorable_type,
                deltas=deltas or {},
            )
            s.add(d)
            s.flush()
            return d
        return self._run(op)

    def get_deltas_for_scorable(self, run_id: str, scorable_id: str) -> List[MetricDeltaORM]:
        def op(s):
            return (
                s.query(MetricDeltaORM)
                .filter_by(run_id=run_id, scorable_id=scorable_id)
                .all()
            )
        return self._run(op)

    # ============================================================
    # VPM OPERATIONS
    # ============================================================

    def save_vpm(
        self,
        run_id: str,
        scorable_id: str,
        scorable_type: str,
        dimension: str,
        width: int,
        height: int,
        image_bytes: bytes,
        meta: dict,
    ) -> MetricVPMORM:
        def op(s):
            vpm = MetricVPMORM(
                run_id=run_id,
                scorable_id=scorable_id,
                scorable_type=scorable_type,
                dimension=dimension,
                width=width,
                height=height,
                image_bytes=image_bytes,
                meta=meta or {},
            )
            s.add(vpm)
            s.flush()
            return vpm
        return self._run(op)

    def get_vpms_for_scorable(self, run_id: str, scorable_id: str) -> List[MetricVPMORM]:
        def op(s):
            return (
                s.query(MetricVPMORM)
                .filter_by(run_id=run_id, scorable_id=scorable_id)
                .all()
            )
        return self._run(op)

    # ============================================================
    # COMBINED FETCHES
    # ============================================================

    def load_all_for_run(self, run_id: str) -> dict:
        """
        Useful for debugging or for visual introspection.
        Returns all vectors + deltas + vpms for a run_id.
        """
        group = self.get_group(run_id)
        if not group:
            return {}

        return {
            "group": group.to_dict(include_children=True),
            "vectors": [v.to_dict() for v in self.get_vectors_for_run(run_id)],
            "deltas":  [d.to_dict() for d in group.deltas],
            "vpms":    [v.to_dict(meta=True) for v in group.vpms],
        }

    def upsert_group_meta(self, run_id: str, patch: dict) -> MetricGroupORM:
        """
        Create the MetricGroup if missing, otherwise shallow-merge `patch` into meta.
        Returns the updated MetricGroupORM row.
        """
        patch = patch or {}
        def op(s):
            g = s.query(MetricGroupORM).filter_by(run_id=run_id).one_or_none()
            if g is None:
                g = MetricGroupORM(run_id=run_id, meta=dict(patch))
                s.add(g)
                s.flush()
                return g
            meta = dict(g.meta or {})
            meta.update(patch)
            g.meta = meta
            s.flush()
            return g
        return self._run(op)

    def get_or_create_group(self, run_id: str, meta: Optional[dict] = None) -> MetricGroupORM:
        """
        Fetch the MetricGroup for run_id, creating it if it doesn't exist.
        If `meta` is provided and the group exists, meta is shallow-merged.
        """
        meta = meta or {}
        def op(s):
            g = s.query(MetricGroupORM).filter_by(run_id=run_id).one_or_none()
            if g is None:
                g = MetricGroupORM(run_id=run_id, meta=dict(meta))
                s.add(g)
                s.flush()
                return g
            if meta:
                current = dict(g.meta or {})
                current.update(meta)
                g.meta = current
                s.flush()
            return g
        return self._run(op)

    def get_group_meta(self, run_id: str) -> dict:
        g = self.get_group(run_id)
        return dict(g.meta or {}) if g else {}

    def get_kept_columns(self, run_id: str) -> list[str]:
        """
        Return the kept feature names selected by MetricFilter for this run.
        Empty list if not present.
        """
        meta = self.get_group_meta(run_id)
        return list((meta.get("metric_filter") or {}).get("kept_columns") or [])

    def get_recent_run_ids(self, limit: int = 5) -> List[str]:
        """
        Return the most recent `run_id`s, newest first.

        Prefers MetricGroupORM (one row per run). If no groups exist, falls back
        to distinct run_ids from MetricVectorORM. Filters out null/empty ids.
        """
        def op(s):
            run_ids: List[str] = []

            # --- Primary: use MetricGroupORM (newest first)
            try:
                # Prefer created_at if available; otherwise order by id desc.
                order_col = getattr(MetricGroupORM, "created_at", None) or MetricGroupORM.id
                q = (
                    s.query(MetricGroupORM.run_id)
                    .order_by(order_col.desc())
                    .limit(limit)
                )
                run_ids = [rid for (rid,) in q.all() if rid]
            except Exception:
                run_ids = []

            # --- Fallback: distinct run_id from MetricVectorORM
            if not run_ids:
                try:
                    order_col_vec = getattr(MetricVectorORM, "created_at", None) or MetricVectorORM.id
                    # DISTINCT ON run_id (portable approach)
                    q2 = (
                        s.query(MetricVectorORM.run_id)
                        .filter(MetricVectorORM.run_id.isnot(None))
                        .filter(MetricVectorORM.run_id != "")
                        .order_by(order_col_vec.desc())
                    )
                    seen = set()
                    for (rid,) in q2.all():
                        if rid not in seen:
                            seen.add(rid)
                            run_ids.append(rid)
                            if len(run_ids) >= limit:
                                break
                except Exception:
                    pass

            return run_ids[:limit]

        return self._run(op)

    def get_run_ids_since(self, since: datetime, limit: Optional[int] = None) -> List[str]:
        """
        Return run_ids whose MetricGroupORM.created_at >= since.
        If `limit` provided, cap results. Newest first.
        """
        def op(s):
            run_ids: List[str] = []
            order_col = getattr(MetricGroupORM, "created_at", None) or MetricGroupORM.id
            q = (
                s.query(MetricGroupORM.run_id)
                .filter(getattr(MetricGroupORM, "created_at", None) >= since)  # if created_at exists
                .order_by(order_col.desc())
            )
            if limit:
                q = q.limit(limit)
            run_ids = [rid for (rid,) in q.all() if rid]
            return run_ids
        return self._run(op)

    def set_kept_columns(self, run_id: str, kept_columns: list[str]) -> MetricGroupORM:
        """
        Persist the feature names selected by MetricFilter / FrontierLens for this run.

        Stored layout in MetricGroupORM.meta:

            {
              "metric_filter": {
                "kept_columns": ["feat1", "feat2", ...]
              },
              ...
            }

        This matches `get_kept_columns` and is safe to call multiple times –
        it will create the MetricGroup if missing, or update the meta if it exists.
        """
        kept_columns = list(dict.fromkeys(kept_columns or []))  # de-dupe, keep order

        def op(s):
            g = (
                s.query(MetricGroupORM)
                .filter_by(run_id=run_id)
                .one_or_none()
            )
            if g is None:
                # Create a new group with minimal meta
                g = MetricGroupORM(run_id=run_id, meta={})
                s.add(g)
                s.flush()

            meta = dict(g.meta or {})
            metric_filter = dict(meta.get("metric_filter") or {})
            metric_filter["kept_columns"] = kept_columns
            meta["metric_filter"] = metric_filter
            g.meta = meta
            s.flush()
            return g

        return self._run(op)
