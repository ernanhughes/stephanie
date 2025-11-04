# stephanie/components/risk/orchestrator.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.risk.epi.epistemic_guard import (GuardInput,
                                                           GuardOutput)
from stephanie.core.manifest import ManifestManager
from stephanie.scoring.scorable import Scorable
from stephanie.services.eg_visual_service import EGVisualService
from stephanie.services.epistemic_guard_service import EpistemicGuardService
from stephanie.services.risk_predictor_service import (RiskPredictorService,
                                                       RiskServiceConfig)
from stephanie.services.scm_service import SCMService
from stephanie.services.storage_service import \
    StorageService  # your FS-backed store
from stephanie.utils.progress_mixin import ProgressMixin

_logger = logging.getLogger(__file__)


@dataclass
class RiskPolicy:
    run_epi_on: Tuple[str, ...] = ("WATCH", "RISK")
    risk_low: float = 0.20
    risk_high: float = 0.60
    visuals_subdir: str = "visuals/risk"
    runs_base_dir: str = "data/risk_runs"


class RiskOrchestrator(ProgressMixin):
    """
    Two-step pipeline (like GAP but for risk):
      1) RiskPredictorService.predict_risk → risk, thresholds, decision/route
      2) EpistemicGuardService.assess (optional) → visuals/evidence stored under run_id
    """

    def __init__(
        self,
        cfg: Dict[str, Any] | None,
        memory,
        container,
        logger,
    ):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        # --- Policy & storage knobs (pulled from agent config if present) ----
        self.policy = RiskPolicy(
            risk_low=float(
                self.cfg.get("thresholds", {}).get("risk_low", 0.20)
            ),
            risk_high=float(
                self.cfg.get("thresholds", {}).get("risk_high", 0.60)
            ),
            visuals_subdir=str(self.cfg.get("visuals_subdir", "visuals/risk")),
            runs_base_dir=str(self.cfg.get("runs_base_dir", "data/risk_runs")),
        )

        # ---- Service registration / initialization --------------------------
        try:
            risk_cfg = RiskServiceConfig(
                bundle_path= "./models/risk/bundle.joblib",
                default_domains=["programming", "llm", "ai", "tech", "general"],
                calib_ttl_s=3600,
                fallback_low=0.20,
                fallback_high=0.60, 
            )
            self.container.register(
                name="risk_predictor_service",
                factory=lambda: RiskPredictorService(risk_cfg, self.memory, self.container, self.logger),
                dependencies=[],
                init_args={
                },
            )
            self.risk_svc = self.container.get("risk_predictor_service")
            self.container.register(
                name="scm_service",
                factory=lambda: SCMService(),
                dependencies=[],
                init_args={"config": cfg.get("scm"), "logger": logger},
            )
            self.scm_service = self.container.get("scm_service")
            self.container.register(
                name="storage",
                factory=lambda: StorageService(),
                dependencies=[],
                init_args={
                    "base_dir": str(self.cfg.get("runs_base_dir") or "data/risk_runs"),
                    "logger": self.logger,
                },
            )
            self.storage = self.container.get("storage")
            self.manifest_manager = ManifestManager(self.storage)
            container.register(
                name="ep_guard",
                factory=lambda: EpistemicGuardService(self.memory, self.container),
                dependencies=[],
                init_args={
                    "config": {
                        "out_dir": f"{self.cfg.get('runs_base_dir')}/eg",
                        "thresholds": (0.2, 0.6),
                    },
                    "logger": logger,
                },
            )
            self.eg_svc = self.container.get("ep_guard")
            container.register(
                name="eg_visual",
                factory=lambda: EGVisualService(),
                dependencies=[],
                init_args={
                    "config": {"out_dir": f"{self.cfg.get('runs_base_dir')}/eg/img"},
                    "logger": logger,
                },
            )
            self.eg_visual_svc = self.container.get("eg_visual")
        except Exception as e:
            _logger.error(f"RiskOrchestrator service registration failed: {e}")

        # progress/manifest
        self._init_progress(container, logger)

        self.logger.info(
            "RiskOrchestrator ready",
            extra={
                "runs_base_dir": self.policy.runs_base_dir,
                "visuals_subdir": self.policy.visuals_subdir,
                "risk_bands": [self.policy.risk_low, self.policy.risk_high],
            },
        )

    # ------------------------------------------------------------------ utils

    # ------------------------------------------------------------ public API
    async def  execute_assessment(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        End-to-end run (mirrors GAP's execute_analysis but for risk).
        Accepts either a single (goal, reply) or a list under context["items"].
        """
        run_id = context.get("pipeline_run_id")

        # manifest start Hey Cortana Hey Cortana I
        m = self.manifest_manager.start_run(
            run_id=run_id,
            dataset=context.get("dataset", "adhoc"),
            models={"chat": context.get("model_alias", "chat")},
        )

        items = context.get("scorables")


        self.pstart(
            task=f"risk:{run_id}",
            total=len(items),
            meta={"mode": "execute_assessment"},
        )


        records = []
        total = len(items)
        _logger.debug(
            f"RiskAgent: evaluating run_id={run_id} models={self.cfg.get('default_model_alias')}"
        )
        for idx, item in enumerate(items):
            scorable = Scorable.from_dict(item)
            goal = item.get("goal_ref") or context.get("goal")

            rec = await self._evaluate_one(
                run_id=run_id,
                goal=goal,
                scorable=scorable,
                model_alias=self.cfg.get("default_model_alias"),
                monitor_alias=self.cfg.get("default_monitor_alias"),
                context=context,
            )
            self.ptick(task=f"risk:{run_id}", done=idx, total=total)
            records.append(rec)

        self.pdone(task=f"risk:{run_id}", extra={"count": len(records)})

        # Build simple summary and persist
        summary = self._summarize(records)
        self.storage.save_json(
            run_id=run_id,
            subdir="results",
            name="risk_summary.json",
            obj=summary,
        )
        result = {
            "run_id": run_id,
            "records": records, 
            "summary": summary,
            "manifest": m.to_dict(),
        }
        self.manifest_manager.finish_run(run_id, result)
        context["risk_assessment"] = result
        return context

    async def _evaluate_one(
        self,
        *,
        scorable: Scorable,
        goal: Dict[str, Any],
        model_alias: str,
        monitor_alias: str,
        run_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Step 1: risk prediction
        risk, (low, high), meta = await self.risk_svc.predict_for_scorable(
            scorable, scorable.text
        )

        decision, route = self._decide(risk, low, high)

        record: Dict[str, Any] = {
            "run_id": run_id,
            "model_alias": model_alias,
            "monitor_alias": monitor_alias,
            "decision": decision,  # OK | WATCH | RISK
            "route": route,  # FAST | MEDIUM | HIGH
            "risk": risk,
            "thresholds": {"low": low, "high": high},
            "metrics": {},
            "artifacts": {},
            "elapsed_ms": None,
        }

        # Step 2: EPI (optional)
        if self.eg_svc and decision in self.policy.run_epi_on:
            gi = GuardInput(
                trace_id=f"{run_id}-{model_alias}-{int(time.time() * 1000) % 10_000_000}",
                question=goal.get("goal_text", goal.get("text","Assess risk")), 
                context=model_alias,
                reference=(context or {}).get("reference", "")
                if context
                else "",
                hypothesis=scorable.text,
                hrm_view=(context or {}).get("hrm_view") if context else None,
                tiny_view=(context or {}).get("tiny_view")
                if context
                else None,
                trust=float((context or {}).get("trust", 0.8)),
                recency=float((context or {}).get("recency", 0.0)),
                meta=(context or {}).get("meta") if context else None,
            )
            try:
                out: GuardOutput = await self.eg_svc.assess(gi, run_id=run_id)
                record["metrics"].update(out.metrics or {})
                record["artifacts"].update(
                    {
                        "vpm": out.vpm_path,
                        "field": out.field_path,
                        "strip": out.strip_path,
                        "legend": out.legend_path,
                        "badge": out.badge_path,
                    }
                )
                # append to per-run index
                if self.storage:
                    await self._upsert_risk_index(
                        run_id,
                        {
                            "trace_id": gi.trace_id,
                            "decision": decision,
                            "risk": risk,
                            "thresholds": {"low": low, "high": high},
                            "metrics": record["metrics"],
                            "paths": record["artifacts"],
                        },
                    )
            except Exception as e:
                self.logger.warning(
                    "EpistemicGuardService.assess failed: %s", e
                )

        record["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return record

    # --------------------------------------------------------------- helpers
    def _decide(self, risk: float, low: float, high: float) -> tuple[str, str]:
        if risk < low:
            return "OK", "FAST"
        if risk < high:
            return "WATCH", "MEDIUM"
        return "RISK", "HIGH"

    async def _upsert_risk_index(
        self, run_id: str, entry: Dict[str, Any]
    ) -> None:
        if not self.storage:
            return
        sub = self.policy.visuals_subdir
        try:
            run_dir = self.storage.subdir(run_id, sub)
            idx_path = run_dir / "risk_index.json"
            if idx_path.exists():
                current = json.loads(idx_path.read_text(encoding="utf-8"))
                if not isinstance(current, list):
                    current = [current]
            else:
                current = []
        except Exception:
            current = []
        current.append(entry)
        self.storage.save_json(
            run_id=run_id, subdir=sub, name="risk_index.json", obj=current
        )

    def _summarize(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(records)
        bands = {"OK": 0, "WATCH": 0, "RISK": 0}
        risks = [r.get("risk", 0.0) for r in records]
        for r in records:
            bands[r.get("decision", "OK")] = (
                bands.get(r.get("decision", "OK"), 0) + 1
            )
        return {
            "count": n,
            "bands": bands,
            "risk_low": self.policy.risk_low,
            "risk_high": self.policy.risk_high,
            "risk_mean": (sum(risks) / n) if n else 0.0,
            "risk_max": max(risks) if risks else 0.0,
        }
