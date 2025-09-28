# stephanie/experiments/engine.py
from __future__ import annotations
from typing import Any, Dict, Optional
from stephanie.utils.json_sanitize import dumps_safe

class ExperimentEngine:
    """
    Small orchestration layer:
    - variant assignment
    - result recording
    - validation + decision
    - optional breadcrumbs to casebooks for dashboards
    """

    def __init__(self, store, memory, logger):
        self.store = store
        self.memory = memory
        self.logger = logger

    def enroll(self, exp_name: str, *, domain: Optional[str], config: Dict[str, Any], group_payloads: Dict[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure experiment+variants exist, assign a variant, and return its payload.
        `group_payloads`: {"A": {...}, "B": {...}} (e.g., Strategy.to_dict())
        """
        exp = self.store.get_or_create_experiment(exp_name, domain=domain, config=config)
        va = self.store.upsert_variant(exp.id, "A", is_control=True,  payload=group_payloads.get("A"))
        vb = self.store.upsert_variant(exp.id, "B", is_control=False, payload=group_payloads.get("B"))

        variant = self.store.assign_variant(exp, case_id=context.get("case_id"), deterministic=True)
        payload = (variant.payload or {})  # your Strategy/Arena settings

        # breadcrumb for dashboards (casebook scorable)
        try:
            case_id = context.get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id,
                    role=f"ab_enroll.{exp_name}",
                    text=dumps_safe({"experiment": exp_name, "group": variant.name, "payload": payload}),
                    pipeline_run_id=context.get("pipeline_run_id"),
                    meta={"group": variant.name, "experiment": exp_name}
                )
        except Exception:
            pass

        return {"group": variant.name, "payload": payload, "variant_id": variant.id, "experiment_id": exp.id}

    def record_result(self, *, variant_id: int, case_id: Optional[int], performance: float, metrics: Dict[str, float], context: Dict[str, Any]) -> None:
        trial = self.store.complete_trial(
            variant_id=variant_id, case_id=case_id,
            performance=performance, metrics=metrics,
            tokens=context.get("tokens"), cost=context.get("cost"), wall_sec=context.get("wall_sec"),
            pipeline_run_id=context.get("pipeline_run_id"), domain=context.get("domain"), meta=context.get("meta")
        )
        # small breadcrumb
        try:
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id,
                    role=f"ab_result.{trial.variant.experiment.name}",
                    text=dumps_safe({"variant": trial.variant.name, "performance": performance, "metrics": metrics}),
                    pipeline_run_id=context.get("pipeline_run_id"),
                    meta={"experiment": trial.variant.experiment.name}
                )
        except Exception:
            pass

    def validate(self, exp_name: str, *, domain: Optional[str], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        exp = self.store.get_or_create_experiment(exp_name, domain=domain, config=config)
        stats = self.store.validate_simple(exp.id, window_seconds=config.get("window_seconds"), min_per_group=config.get("min_samples_per_group", 8))
        if not stats:
            return None

        # decision heuristic (mirrors your current thresholds)
        min_rel_impr = float(config.get("min_strategy_improvement", 0.02))
        max_p = float(config.get("max_p_value", 0.10))
        recommend = (stats["delta_B_minus_A"] > 0.0) and (stats["relative_improvement"] >= min_rel_impr) and ((stats["p_value_welch"] <= max_p) or (stats["p_value_mann_whitney"] <= max_p))

        out = {**stats, "recommend_commit": bool(recommend)}
        return out
