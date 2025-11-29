# stephanie/memory/metric_store.py
from __future__ import annotations

import datetime
from typing import List, Optional, Dict, Any, Tuple

from sqlalchemy import desc, func

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.metrics import (
    MetricDeltaORM, 
    MetricGroupORM,
    MetricVectorORM, 
    MetricVPMORM,
    CriticRunORM,  # NEW
    CriticModelORM  # NEW
)


class MetricStore(BaseSQLAlchemyStore):
    """
    Enhanced MetricStore with Frontier Intelligence support:
      - MetricGroupORM (pipeline runs)
      - MetricVectorORM (metrics per scorable)
      - MetricDeltaORM (diffs)
      - MetricVPMORM (images)
      - CriticRunORM (critic decision history)
      - CriticModelORM (model version tracking)
    """
    
    orm_model = MetricGroupORM
    default_order_by = "created_at"

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "metrics"

    # ============================================================
    # FRONTIER INTELLIGENCE OPERATIONS
    # ============================================================
    
    def record_critic_run(
        self,
        run_id: str,
        model_version: str,
        auc: float,
        band_separation: float = 0.0,
        stability_score: float = 0.0,
        feature_consistency: float = 0.0,
        is_promoted: bool = False,
        decision_action: str = "WAIT",
        decision_confidence: float = 0.0,
        decision_reason: str = "",
        decision_advice: str = ""
    ) -> CriticRunORM:
        """Record a critic run with detailed metrics and decisions"""
        def op(s):
            # Ensure the metric group exists
            group = s.query(MetricGroupORM).filter_by(run_id=run_id).one_or_none()
            if not group:
                group = MetricGroupORM(run_id=run_id)
                s.add(group)
                s.flush()
            
            # Create the critic run record
            critic_run = CriticRunORM(
                run_id=run_id,
                model_version=model_version,
                auc=auc,
                band_separation=band_separation,
                stability_score=stability_score,
                feature_consistency=feature_consistency,
                is_promoted=is_promoted,
                decision_action=decision_action,
                decision_confidence=decision_confidence,
                decision_reason=decision_reason,
                decision_advice=decision_advice
            )
            
            s.add(critic_run)
            s.flush()
            return critic_run
        return self._run(op)
    
    def get_critic_runs(
        self,
        run_id: Optional[str] = None,
        model_version: Optional[str] = None,
        limit: int = 100
    ) -> List[CriticRunORM]:
        """Get critic runs with optional filtering"""
        def op(s):
            query = s.query(CriticRunORM)
            
            if run_id:
                query = query.filter(CriticRunORM.run_id == run_id)
            if model_version:
                query = query.filter(CriticRunORM.model_version == model_version)
            
            return query.order_by(CriticRunORM.created_at.desc()).limit(limit).all()
        return self._run(op)
    
    def get_critic_trend_data(
        self,
        run_id: Optional[str] = None,
        window: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get trend data for critic metrics over recent runs
        
        Returns:
            {
                "auc": [{"value": 0.85, "timestamp": "..."}, ...],
                "band_separation": [...],
                ...
            }
        """
        def op(s):
            query = s.query(CriticRunORM).order_by(CriticRunORM.created_at.desc())
            
            if run_id:
                query = query.filter(CriticRunORM.run_id == run_id)
            
            runs = query.limit(window).all()
            
            # Build trend data
            trend_data = {
                "auc": [],
                "band_separation": [],
                "stability_score": [],
                "feature_consistency": [],
                "decision_action": []
            }
            
            for run in reversed(runs):  # Reverse to get chronological order
                timestamp = run.created_at.isoformat()
                trend_data["auc"].append({"value": run.auc, "timestamp": timestamp})
                trend_data["band_separation"].append({"value": run.band_separation, "timestamp": timestamp})
                trend_data["stability_score"].append({"value": run.stability_score, "timestamp": timestamp})
                trend_data["feature_consistency"].append({"value": run.feature_consistency, "timestamp": timestamp})
                trend_data["decision_action"].append({"value": run.decision_action, "timestamp": timestamp})
            
            return trend_data
        return self._run(op)
    
    def update_critic_model(
        self,
        model_version: str,
        run_id: str,
        auc: float,
        band_separation: float = 0.0,
        stability_score: float = 0.0,
        feature_consistency: float = 0.0,
        is_active: bool = False,
        is_best: bool = False,
        promoted_at: Optional[datetime] = None
    ) -> CriticModelORM:
        """Update or create a critic model record"""
        def op(s):
            # Check if model exists
            model = s.query(CriticModelORM).filter_by(model_version=model_version).one_or_none()
            
            if model:
                # Update existing model
                model.run_id = run_id
                model.auc = auc
                model.band_separation = band_separation
                model.stability_score = stability_score
                model.feature_consistency = feature_consistency
                model.is_active = is_active
                model.is_best = is_best
                if is_active and not model.promoted_at:
                    model.promoted_at = datetime.now()
                elif not is_active:
                    model.promoted_at = None
            else:
                # Create new model
                model = CriticModelORM(
                    model_version=model_version,
                    run_id=run_id,
                    auc=auc,
                    band_separation=band_separation,
                    stability_score=stability_score,
                    feature_consistency=feature_consistency,
                    is_active=is_active,
                    is_best=is_best,
                    promoted_at=promoted_at or (datetime.now() if is_active else None)
                )
                s.add(model)
            
            s.flush()
            return model
        return self._run(op)
    
    def get_best_critic_model(self) -> Optional[CriticModelORM]:
        """Get the best critic model based on AUC and stability"""
        def op(s):
            return s.query(CriticModelORM).filter_by(is_best=True).first()
        return self._run(op)
    
    def get_active_critic_model(self) -> Optional[CriticModelORM]:
        """Get the currently active critic model"""
        def op(s):
            return s.query(CriticModelORM).filter_by(is_active=True).first()
        return self._run(op)
    
    def get_critic_decision_history(
        self,
        run_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get the decision history for the critic"""
        def op(s):
            query = s.query(
                CriticRunORM.run_id,
                CriticRunORM.model_version,
                CriticRunORM.auc,
                CriticRunORM.decision_action,
                CriticRunORM.decision_confidence,
                CriticRunORM.decision_reason,
                CriticRunORM.created_at
            ).order_by(CriticRunORM.created_at.desc())
            
            if run_id:
                query = query.filter(CriticRunORM.run_id == run_id)
            
            results = query.limit(limit).all()
            
            return [{
                "run_id": r[0],
                "model_version": r[1],
                "auc": r[2],
                "decision_action": r[3],
                "decision_confidence": r[4],
                "decision_reason": r[5],
                "created_at": r[6].isoformat() if r[6] else None
            } for r in results]
        return self._run(op)
    
    def get_critic_progress_report(self) -> Dict[str, Any]:
        """Get a comprehensive progress report for the critic system"""
        def op(s):
            # Get latest critic run
            latest_run = s.query(CriticRunORM).order_by(CriticRunORM.created_at.desc()).first()
            
            # Get best model
            best_model = s.query(CriticModelORM).filter_by(is_best=True).first()
            
            # Get active model
            active_model = s.query(CriticModelORM).filter_by(is_active=True).first()
            
            # Get recent decision history
            decision_history = self.get_critic_decision_history(limit=5)
            
            # Calculate trends
            trend_data = self.get_critic_trend_data(window=5)
            
            # Determine overall status
            status = "STABLE"
            if trend_data["auc"] and len(trend_data["auc"]) >= 2:
                auc_trend = trend_data["auc"][-1]["value"] - trend_data["auc"][0]["value"]
                if auc_trend > 0.01:
                    status = "IMPROVING"
                elif auc_trend < -0.01:
                    status = "DEGRADING"
            
            return {
                "status": status,
                "latest_run": latest_run.to_dict() if latest_run else None,
                "best_model": best_model.to_dict() if best_model else None,
                "active_model": active_model.to_dict() if active_model else None,
                "decision_history": decision_history,
                "trend_data": trend_data,
                "timestamp": datetime.now().isoformat()
            }
        return self._run(op)
    
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
    
    def update_group_with_critic_data(
        self,
        run_id: str,
        frontier_metric: Optional[str] = None,
        critic_status: Optional[str] = None,
        critic_action: Optional[str] = None,
        critic_confidence: Optional[float] = None,
        is_best_model: Optional[bool] = None,
        model_version: Optional[str] = None,
        auc_score: Optional[float] = None,
        band_separation: Optional[float] = None,
        stability_score: Optional[float] = None,
        feature_consistency: Optional[float] = None
    ) -> MetricGroupORM:
        """Update a metric group with critic-related data"""
        def op(s):
            g = s.query(MetricGroupORM).filter_by(run_id=run_id).one_or_none()
            if not g:
                g = MetricGroupORM(run_id=run_id)
                s.add(g)
            
            # Update fields if provided
            if frontier_metric is not None:
                g.frontier_metric = frontier_metric
            if critic_status is not None:
                g.critic_status = critic_status
            if critic_action is not None:
                g.critic_action = critic_action
            if critic_confidence is not None:
                g.critic_confidence = critic_confidence
            if is_best_model is not None:
                g.is_best_model = is_best_model
            if model_version is not None:
                g.model_version = model_version
            if auc_score is not None:
                g.auc_score = auc_score
            if band_separation is not None:
                g.band_separation = band_separation
            if stability_score is not None:
                g.stability_score = stability_score
            if feature_consistency is not None:
                g.feature_consistency = feature_consistency
            
            s.flush()
            return g
        return self._run(op)

    def get_group(self, run_id: str) -> Optional[MetricGroupORM]:
        def op(s):
            return s.query(MetricGroupORM).filter_by(run_id=run_id).one_or_none()
        return self._run(op)

    # ============================================================
    # EXISTENCE CHECKS BY SCORABLE
    # ============================================================

    def vector_exists_by_scorable(self, scorable_id: str, scorable_type: str) -> bool:
        def op(s):
            q = (s.query(MetricVectorORM.id)
                   .filter(MetricVectorORM.scorable_id == scorable_id,
                           MetricVectorORM.scorable_type == scorable_type)
                   .limit(1))
            return s.query(q.exists()).scalar()
        return self._run(op)

    # ============================================================
    # INSERT-OR-UPDATE BY SCORABLE
    # ============================================================

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

    # ============================================================
    # "SKIP IF EXISTS" READ CHECK FOR FAST SHORT-CIRCUITING
    # ============================================================

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