# stephanie/components/risk/orchestrator.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.core.manifest import ManifestManager
from stephanie.utils.progress_mixin import ProgressMixin
from stephanie.services.risk_predictor_service import RiskPredictorService, RiskServiceConfig
from stephanie.services.epistemic_guard_service import EpistemicGuardService
from stephanie.components.risk.epi.epistemic_guard import GuardInput, GuardOutput

from stephanie.services.storage_service import StorageService  # your FS-backed store

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
            risk_low=float(self.cfg.get("thresholds", {}).get("risk_low", 0.20)),
            risk_high=float(self.cfg.get("thresholds", {}).get("risk_high", 0.60)),
            visuals_subdir=str(self.cfg.get("visuals_subdir", "visuals/risk")),
            runs_base_dir=str(self.cfg.get("runs_base_dir", "data/risk_runs")),
        )

        # ---- Service registration / initialization --------------------------
        self.storage = self._ensure_storage()
        self.manifest_manager = ManifestManager(self.storage)
        self.risk_svc = self._ensure_risk_service()
        self.eg_svc = self._ensure_eg_service()

        # progress/manifest
        self._init_progress(container, logger)

        self.logger.info(
            "RiskOrchestrator ready",
            extra={
                "runs_base_dir": self.policy.runs_base_dir,
                "visuals_subdir": self.policy.visuals_subdir,
                "risk_bands": [self.policy.risk_low, self.policy.risk_high],
                "epi_enabled": bool(self.eg_svc),
            },
        )

    # ------------------------------------------------------------------ utils
    def _pick(self, keys: List[str], default: Any) -> Any:
        """
        Read nested keys from cfg dict (supports dotted 'a.b.c' under top-level risk block).
        """
        # allow either top-level or nested under 'risk'
        roots = [self.cfg, self.cfg.get("risk", {})] if isinstance(self.cfg, dict) else [self.cfg]
        for root in roots:
            cur = root
            ok = True
            for part in keys[0].split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok:
                return cur
        return default

    def _ensure_storage(self):
        # Prefer an existing storage on the container
        st = getattr(self.container, "storage", None) or getattr(self.container, "storage", None)
        if st:
            return st

        if StorageService is None:
            self.logger.warning("No GapStorageService available; run-scoped writes may fail.")
            return None

        st = StorageService()
        st.initialize(base_dir=self.policy.runs_base_dir)
        # Set on container for reuse
        try:
            setattr(self.container, "storage", st)
        except Exception:
            pass
        return st

    def _ensure_risk_service(self) -> RiskPredictorService:
        svc = getattr(self.container, "risk_predictor_service", None)
        if svc:
            return svc

        # Compose service config from agent cfg if present
        svc_cfg = RiskServiceConfig(
            bundle_path=str(self._pick(["bundle_path"], "./models/risk/bundle.joblib")),
            default_domains=tuple(self._pick(["default_domains"], ["science", "history", "geography", "tech", "general"])),
            calib_ttl_s=int(self._pick(["calib_ttl_s"], 3600)),
            fallback_low=float(self._pick(["fallback_low"], 0.20)),
            fallback_high=float(self._pick(["fallback_high"], 0.60)),
        )
        svc = RiskPredictorService(svc_cfg, memory=self.memory, logger=self.logger)
        svc.initialize(config=svc_cfg.__dict__)
        try:
            setattr(self.container, "risk_predictor_service", svc)
        except Exception:
            pass
        return svc

    def _ensure_eg_service(self) -> Optional[EpistemicGuardService]:
        svc = getattr(self.container, "epistemic_guard_service", None)
        if svc:
            return svc
        try:
            svc = EpistemicGuardService()
            svc.initialize(
                config={
                    "visuals_subdir": self.policy.visuals_subdir,
                    "thresholds": (self.policy.risk_low, self.policy.risk_high),
                    "seed": int(self._pick(["seed"], 42)),
                    # Optional static out_dir if no storage given:
                    "out_dir": f"{self.policy.runs_base_dir}/adhoc/{self.policy.visuals_subdir}",
                },
                storage=self.storage,
                logger=self.logger,
            )
            try:
                setattr(self.container, "epistemic_guard_service", svc)
            except Exception:
                pass
            return svc
        except Exception as e:
            self.logger.warning(f"EpistemicGuardService could not be initialized: {e}")
            return None

    # ------------------------------------------------------------ public API
    async def execute_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        End-to-end run (mirrors GAP's execute_analysis but for risk).
        Accepts either a single (goal, reply) or a list under context["items"].
        """
        run_id = (
            context.get("pipeline_run_id")
            or context.get("run_id")
            or "risk_run"
        )

        # manifest start
        m = self.manifest_manager.start_run(
            run_id=run_id,
            dataset=context.get("dataset", "adhoc"),
            models={"chat": context.get("model_alias", "chat")},
        )

        items: List[Dict[str, Any]]
        if "items" in context and isinstance(context["items"], list):
            items = context["items"]
        else:
            items = [{
                "goal": context.get("goal", ""),
                "reply": context.get("reply", ""),
                "model_alias": context.get("model_alias", self._pick(["default_model_alias"], "chat")),
                "monitor_alias": context.get("monitor_alias", self._pick(["default_monitor_alias"], "tiny")),
                "context": context.get("context", {}),
            }]

        self.pstart(task=f"risk:{run_id}", total=len(items), meta={"mode": "execute_assessment"})

        records: List[Dict[str, Any]] = []
        for idx, it in enumerate(items, 1):
            rec = await self._evaluate_one(
                goal=it.get("goal", ""),
                reply=it.get("reply", ""),
                model_alias=it.get("model_alias", "chat"),
                monitor_alias=it.get("monitor_alias", "tiny"),
                run_id=run_id,
                context=it.get("context", {}),
            )
            records.append(rec)
            self.ptick(task=f"risk:{run_id}", n=1)

        self.pdone(task=f"risk:{run_id}", extra={"count": len(records)})

        # Build simple summary and persist
        summary = self._summarize(records)
        if self.storage:
            self.storage.save_json(run_id=run_id, subdir="results", name="risk_summary.json", obj=summary)

        result = {
            "run_id": run_id,
            "records": records,
            "summary": summary,
            "manifest": m.to_dict(),
        }
        self.manifest_manager.finish_run(run_id, result)
        return result

    async def _evaluate_one(
        self,
        *,
        goal: str,
        reply: str,
        model_alias: str,
        monitor_alias: str,
        run_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Step 1: risk prediction
        risk, (low, high) = await self.risk_svc.predict_risk(goal, reply)
        decision, route = self._decide(risk, low, high)

        record: Dict[str, Any] = {
            "run_id": run_id,
            "model_alias": model_alias,
            "monitor_alias": monitor_alias,
            "decision": decision,      # OK | WATCH | RISK
            "route": route,            # FAST | MEDIUM | HIGH
            "risk": risk,
            "thresholds": {"low": low, "high": high},
            "metrics": {},
            "artifacts": {},
            "elapsed_ms": None,
        }

        # Step 2: EPI (optional)
        if self.eg_svc and decision in self.policy.run_epi_on:
            gi = GuardInput(
                trace_id=f"{run_id}-{model_alias}-{int(time.time()*1000)%10_000_000}",
                question=goal,
                context=model_alias,
                reference=(context or {}).get("reference", "") if context else "",
                hypothesis=reply,
                hrm_view=(context or {}).get("hrm_view") if context else None,
                tiny_view=(context or {}).get("tiny_view") if context else None,
                trust=float((context or {}).get("trust", 0.8)),
                recency=float((context or {}).get("recency", 0.0)),
                meta=(context or {}).get("meta") if context else None,
            )
            try:
                out: GuardOutput = await self.eg_svc.assess(gi, run_id=run_id)
                record["metrics"].update(out.metrics or {})
                record["artifacts"].update({
                    "vpm": out.vpm_path,
                    "field": out.field_path,
                    "strip": out.strip_path,
                    "legend": out.legend_path,
                    "badge": out.badge_path,
                })
                # append to per-run index
                if self.storage:
                    await self._upsert_risk_index(run_id, {
                        "trace_id": gi.trace_id,
                        "decision": decision,
                        "risk": risk,
                        "thresholds": {"low": low, "high": high},
                        "metrics": record["metrics"],
                        "paths": record["artifacts"],
                    })
            except Exception as e:
                self.logger.warning("EpistemicGuardService.assess failed: %s", e)

        record["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return record

    # --------------------------------------------------------------- helpers
    def _decide(self, risk: float, low: float, high: float) -> tuple[str, str]:
        if risk < low:      return "OK", "FAST"
        if risk < high:     return "WATCH", "MEDIUM"
        return "RISK", "HIGH"

    async def _upsert_risk_index(self, run_id: str, entry: Dict[str, Any]) -> None:
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
        self.storage.save_json(run_id=run_id, subdir=sub, name="risk_index.json", obj=current)

    def _summarize(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(records)
        bands = {"OK":0, "WATCH":0, "RISK":0}
        risks = [r.get("risk", 0.0) for r in records]
        for r in records:
            bands[r.get("decision","OK")] = bands.get(r.get("decision","OK"), 0) + 1
        return {
            "count": n,
            "bands": bands,
            "risk_low": self.policy.risk_low,
            "risk_high": self.policy.risk_high,
            "risk_mean": (sum(risks)/n) if n else 0.0,
            "risk_max": max(risks) if risks else 0.0,
        }
