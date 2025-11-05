# stephanie/stores/experiments_store.py
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from stephanie.memory.base_store import BaseSQLAlchemyStore
# NOTE: your models live under singular "experiment" per your snippet
from stephanie.models.experiment import (ExperimentModelSnapshotORM,
                                         ExperimentORM, TrialMetricORM,
                                         TrialORM, VariantORM)
from stephanie.utils.statistics_utils import (bootstrap_ci, cohens_d,
                                              mann_whitney_u, mean_stdev,
                                              welch_ttest, winsorize)


class ExperimentStore(BaseSQLAlchemyStore):
    """
    ORM-backed repository for A/B experiments.
    Mirrors EvaluationStore ergonomics: inherits BaseSQLAlchemyStore,
    uses _run() and exposes small, composable methods.
    """

    orm_model = ExperimentORM
    # Keep a default ordering compatible with BaseSQLAlchemyStore conventions
    default_order_by = ExperimentORM.created_at.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "experiments"
        self.table_name = "experiments"

    # ---------------------------------------------------------------------
    # Experiments
    # ---------------------------------------------------------------------
    def get_or_create_experiment(
        self,
        name: str,
        *,
        domain: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
    ) -> ExperimentORM:
        def op(s):
            q = s.query(ExperimentORM).filter(ExperimentORM.name == name)
            if domain:
                q = q.filter(ExperimentORM.domain == domain)
            exp = q.one_or_none()
            if exp:
                return exp
            exp = ExperimentORM(
                name=name,
                domain=domain,
                config=(config or {}),
                label=(label or None),
                status="active",
            )
            s.add(exp)
            s.flush()
            return exp

        return self._run(op)

    def get_by_name(self, name: str, *, domain: Optional[str] = None) -> Optional[dict]:
        def op(s):
            q = s.query(ExperimentORM).filter(ExperimentORM.name == name)
            if domain:
                q = q.filter(ExperimentORM.domain == domain)
            row = q.one_or_none()
            return self._orm_to_dict_experiment(row) if row else None

        return self._run(op)

    def list(self, limit: int = 100) -> List[dict]:
        def op(s):
            q = s.query(ExperimentORM).order_by(self.default_order_by).limit(limit)
            return [self._orm_to_dict_experiment(r) for r in q.all()]

        return self._run(op)

    # ---------------------------------------------------------------------
    # Variants
    # ---------------------------------------------------------------------
    def upsert_variant(
        self,
        experiment_id: int,
        name: str,
        *,
        is_control: bool = False,
        payload: Optional[Dict[str, Any]] = None,
    ) -> VariantORM:
        def op(s):
            v = (
                s.query(VariantORM)
                .filter(
                    VariantORM.experiment_id == experiment_id,
                    VariantORM.name == name,
                )
                .one_or_none()
            )
            if v:
                if payload is not None:
                    v.payload = payload
                v.is_control = bool(is_control)
                s.flush()
                return v
            v = VariantORM(
                experiment_id=experiment_id,
                name=name,
                is_control=is_control,
                payload=(payload or {}),
            )
            s.add(v)
            s.flush()
            return v

        return self._run(op)

    def list_variants(self, experiment_id: int) -> List[dict]:
        def op(s):
            q = s.query(VariantORM).filter(VariantORM.experiment_id == experiment_id)
            return [self._orm_to_dict_variant(v) for v in q.all()]

        return self._run(op)

    # ---------------------------------------------------------------------
    # Assignment
    # ---------------------------------------------------------------------
    def assign_variant(
        self,
        experiment: ExperimentORM,
        *,
        case_id: Optional[int],
        deterministic: bool = True,
    ) -> VariantORM:
        """
        Deterministic by default: hash((case_id, experiment.name)) → stable arm.
        If case_id is None, pick "B" if present, else the first control/variant.
        Also creates a Trial shell so we don’t double-assign later.
        """
        def op(s):
            exp = s.query(ExperimentORM).get(experiment.id)
            if not exp:
                raise ValueError("Experiment not found")

            variants = sorted(exp.variants, key=lambda v: (not v.is_control, v.name))
            if not variants:
                raise ValueError("No variants registered for experiment")

            if deterministic and case_id is not None:
                idx = (hash((case_id, exp.name)) & 0x7FFFFFFF) % max(1, len(variants))
                pick = variants[idx]
            else:
                named = {v.name.upper(): v for v in variants}
                pick = named.get("B") or variants[0]

            # Ensure a trial shell exists
            trial = (
                s.query(TrialORM)
                .filter(TrialORM.variant_id == pick.id, TrialORM.case_id == case_id)
                .one_or_none()
            )
            if not trial:
                trial = TrialORM(
                    variant_id=pick.id,
                    case_id=case_id,
                    assigned_at=datetime.now(),
                )
                s.add(trial)
                s.flush()
            s.refresh(pick)
            return pick

        return self._run(op)

    # ---------------------------------------------------------------------
    # Recording results
    # ---------------------------------------------------------------------
    def complete_trial(
        self,
        *,
        variant_id: int,
        case_id: Optional[int],
        performance: Optional[float],
        metrics: Optional[Dict[str, float]] = None,
        tokens: Optional[int] = None,
        cost: Optional[float] = None,
        wall_sec: Optional[float] = None,
        pipeline_run_id: Optional[str] = None,
        domain: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        experiment_group: Optional[str] = None,      
        tags_used: Optional[List[str]] = None,       
    ) -> TrialORM:
        def op(s: Session):
            trial = (
                s.query(TrialORM)
                .filter(TrialORM.variant_id == variant_id, TrialORM.case_id == case_id)
                .one_or_none()
            )
            if not trial:
                trial = TrialORM(
                    variant_id=variant_id,
                    case_id=case_id,
                    assigned_at=datetime.now(),
                )
                s.add(trial)
                s.flush()

            trial.performance = performance
            trial.completed_at = datetime.now()
            trial.tokens = tokens
            trial.cost = cost
            trial.wall_sec = wall_sec
            trial.pipeline_run_id = pipeline_run_id
            trial.domain = domain
            trial.meta = (meta or {})
            trial.experiment_group = experiment_group
            trial.tags_used = tags_used or []
            s.add(trial)
            s.flush()

            for k, v in (metrics or {}).items():
                s.add(TrialMetricORM(trial_id=trial.id, key=str(k), value=float(v)))
            s.flush()
            return trial

        return self._run(op)

    # ---------------------------------------------------------------------
    # Reads for validation & dashboards
    # ---------------------------------------------------------------------
    def recent_trials(
        self,
        experiment_id: int,
        *,
        window_seconds: Optional[int] = None,
        limit: int = 400,
    ) -> List[Dict[str, Any]]:
        def op(s):
            q = (
                s.query(TrialORM, VariantORM)
                .join(VariantORM, VariantORM.id == TrialORM.variant_id)
                .filter(VariantORM.experiment_id == experiment_id)
                .order_by(TrialORM.completed_at.desc().nullslast())
            )
            if window_seconds:
                cutoff = datetime.now() - timedelta(seconds=window_seconds)
                q = q.filter(TrialORM.assigned_at >= cutoff)

            rows = q.limit(limit).all()
            out: List[Dict[str, Any]] = []
            for trial, var in rows:
                out.append(
                    {
                        "group": var.name,
                        "performance": trial.performance,
                        "domain": trial.domain or "default",
                        "tokens": trial.tokens,
                        "cost": trial.cost,
                        "wall_sec": trial.wall_sec,
                        "ts": (
                            time.time()
                            if not trial.completed_at
                            else trial.completed_at.timestamp()
                        ),
                        "case_id": trial.case_id,
                    }
                )
            return out

        return self._run(op)

    def validate_simple(
        self,
        experiment_id: int,
        *,
        window_seconds: Optional[int] = None,
        min_per_group: int = 8,
    ) -> Optional[Dict[str, Any]]:
        tests = self.recent_trials(
            experiment_id, window_seconds=window_seconds, limit=400
        )
        a_vals = [
            t for t in tests if t.get("group") == "A" and t.get("performance") is not None
        ]
        b_vals = [
            t for t in tests if t.get("group") == "B" and t.get("performance") is not None
        ]
        if len(a_vals) < min_per_group or len(b_vals) < min_per_group:
            return None

        perf_a = winsorize([float(t["performance"]) for t in a_vals], 0.05)
        perf_b = winsorize([float(t["performance"]) for t in b_vals], 0.05)

        mean_a, sd_a = mean_stdev(perf_a)
        mean_b, sd_b = mean_stdev(perf_b)
        delta = mean_b - mean_a
        rel = (delta / abs(mean_a)) if mean_a != 0 else 0.0

        t, p_t = welch_ttest(perf_a, perf_b)
        u, p_u = mann_whitney_u(perf_a, perf_b)
        d_lo, d_hi, r_lo, r_hi = bootstrap_ci(perf_a, perf_b, iters=2000)
        d = cohens_d(perf_a, perf_b)

        return {
            "samples_A": len(perf_a),
            "samples_B": len(perf_b),
            "mean_A": mean_a,
            "sd_A": sd_a,
            "mean_B": mean_b,
            "sd_B": sd_b,
            "delta_B_minus_A": delta,
            "relative_improvement": rel,
            "cohens_d": d,
            "welch_t": t,
            "p_value_welch": p_t,
            "mann_whitney_u": u,
            "p_value_mann_whitney": p_u,
            "delta_ci95": [d_lo, d_hi],
            "rel_improvement_ci95": [r_lo, r_hi],
        }

    # ---------------------------------------------------------------------
    # Helpers (dict views)
    # ---------------------------------------------------------------------
    def _orm_to_dict_experiment(self, row: ExperimentORM) -> dict:
        if not row:
            return {}
        return {
            "id": row.id,
            "name": row.name,
            "label": row.label,
            "status": row.status,
            "domain": row.domain,
            "config": row.config or {},
            "created_at": row.created_at,
        }

    def _orm_to_dict_variant(self, row: VariantORM) -> dict:
        if not row:
            return {}
        return {
            "id": row.id,
            "experiment_id": row.experiment_id,
            "name": row.name,
            "is_control": bool(row.is_control),
            "payload": row.payload or {},
            "created_at": row.created_at,
        }

    def _orm_to_dict_trial(self, row: TrialORM) -> dict:
        if not row:
            return {}
        return {
            "id": row.id,
            "variant_id": row.variant_id,
            "case_id": row.case_id,
            "pipeline_run_id": row.pipeline_run_id,
            "domain": row.domain,
            "assigned_at": row.assigned_at,
            "completed_at": row.completed_at,
            "performance": row.performance,
            "tokens": row.tokens,
            "cost": row.cost,
            "wall_sec": row.wall_sec,
            "meta": row.meta or {},
        }

    def _orm_to_dict_metric(self, row: TrialMetricORM) -> dict:
        if not row:
            return {}
        return {
            "id": row.id,
            "trial_id": row.trial_id,
            "key": row.key,
            "value": row.value,
            "created_at": row.created_at,
        }


    def save_model_snapshot(
        self,
        *,
        experiment_id: int,
        name: str,
        version: int,
        payload: Dict[str, Any],
        validation: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        committed_at: Optional[datetime] = None,
    ) -> ExperimentModelSnapshotORM:
        """Persist a versioned model snapshot tied to an experiment (commit point)."""

        def op(s):
            row = ExperimentModelSnapshotORM(
                experiment_id=experiment_id,
                name=name,
                domain=domain,
                version=int(version),
                payload=payload or {},
                validation=validation or {},
                committed_at=committed_at or datetime.now(),
                created_at=datetime.now(),
            )
            s.add(row)
            s.flush()
            return row

        return self._run(op)

    def get_latest_model_snapshot(
        self,
        *,
        experiment_name: str,
        name: str,
        domain: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch the latest snapshot payload (and metadata) for a given logical model name."""

        def op(s):
            exp = (
                s.query(ExperimentORM)
                .filter(ExperimentORM.name == experiment_name)
                .filter(ExperimentORM.domain == domain if domain else True)
                .order_by(ExperimentORM.created_at.desc())
                .first()
            )
            if not exp:
                return None

            q = (
                s.query(ExperimentModelSnapshotORM)
                .filter(ExperimentModelSnapshotORM.experiment_id == exp.id)
                .filter(ExperimentModelSnapshotORM.name == name)
            )
            if domain:
                q = q.filter(ExperimentModelSnapshotORM.domain == domain)

            row = q.order_by(ExperimentModelSnapshotORM.version.desc()).first()
            if not row:
                return None

            return {
                "experiment_id": row.experiment_id,
                "name": row.name,
                "domain": row.domain,
                "version": row.version,
                "payload": row.payload,
                "validation": row.validation,
                "committed_at": row.committed_at,
            }

        return self._run(op)


    def validate_groups(
        self,
        experiment_id: int,
        *,
        groups: Optional[List[str]] = None,
        window_seconds: Optional[int] = None,
        min_per_group: int = 8,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare multiple groups (A/B/C...) using ANOVA + pairwise stats.
        Returns dict of group stats + comparisons.
        """
        trials = self.recent_trials(experiment_id, window_seconds=window_seconds, limit=1000)
        if groups:
            trials = [t for t in trials if t.get("experiment_group") in groups]

        grouped: Dict[str, List[float]] = {}
        for t in trials:
            grp = t.get("experiment_group") or t.get("group")  # fallback to A/B
            if t.get("performance") is None:
                continue
            grouped.setdefault(grp, []).append(float(t["performance"]))

        # ensure min samples
        for g, vals in grouped.items():
            if len(vals) < min_per_group:
                return None

        stats = {}
        for g, vals in grouped.items():
            mean, sd = mean_stdev(vals)
            stats[g] = {"n": len(vals), "mean": mean, "sd": sd}

        # pairwise comparisons
        comparisons = {}
        groups_list = list(grouped.keys())
        for i in range(len(groups_list)):
            for j in range(i + 1, len(groups_list)):
                g1, g2 = groups_list[i], groups_list[j]
                v1, v2 = grouped[g1], grouped[g2]
                t, p_t = welch_ttest(v1, v2)
                u, p_u = mann_whitney_u(v1, v2)
                d = cohens_d(v1, v2)
                comparisons[f"{g2}_minus_{g1}"] = {
                    "delta": (stats[g2]["mean"] - stats[g1]["mean"]),
                    "cohens_d": d,
                    "welch_t": t,
                    "p_value_welch": p_t,
                    "mann_whitney_u": u,
                    "p_value_mann_whitney": p_u,
                }

        return {"groups": stats, "comparisons": comparisons}
