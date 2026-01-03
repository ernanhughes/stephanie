from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
# Model-layer imports (pure, no infra)
from stephanie.model.risk_predictor import (DomainCalibratedRiskPredictor,
                                            RiskFeaturizer)
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer

RISK_DIMENSION_NAME = "risk"


class RiskScorer(BaseScorer):
    """
    Domain-calibrated risk scorer.

    This wraps the DomainCalibratedRiskPredictor behind the scorer interface so
    GAP can treat 'risk' as just another dimension. It returns a ScoreBundle with:
      - dimension = "risk"
      - score     = risk probability in [0,1]
      - attributes: domain, thresholds, band, and aligned vector for VPM/SCM

    Config (example):
        {
          "model_path": "models/risk/bundle.joblib",
          "default_low": 0.2,
          "default_high": 0.6,
          "use_service": true,                 # prefer container service if available
          "attr_level": "standard",            # reserved for future verbosity control
          "domain_override": null              # force domain if provided
        }
    """

    def __init__(self, cfg: dict, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_type = "risk"
        self.attr_level = (cfg.get("attr_level") or "standard").lower()
        self.bundle_path: Optional[str] = cfg.get("model_path")
        self.default_low: float = float(cfg.get("default_low", 0.2))
        self.default_high: float = float(cfg.get("default_high", 0.6))
        self.use_service: bool = bool(cfg.get("use_service", True))
        self.domain_override: Optional[str] = cfg.get("domain_override")

        self._service = None
        self._predictor: Optional[DomainCalibratedRiskPredictor] = None

        # Try to bind the service (has async predict_risk + domain/thresholds)
        if self.use_service:
            try:
                self._service = container.get("risk_predictor")
                self.logger.log("RiskScorerServiceAttached", {"ok": True})
            except Exception as e:
                self.logger.log("RiskScorerServiceMissing", {"ok": False, "error": str(e)})
                self._service = None

        # If no service, set up a local predictor with default providers
        if not self._service:
            self._predictor = DomainCalibratedRiskPredictor(
                bundle_path=self.bundle_path,
                featurizer=RiskFeaturizer(),
                default_thresholds=(self.default_low, self.default_high),
            )
            self.logger.log("RiskScorerLocalPredictorReady", {"bundle": bool(self.bundle_path)})

        # Expose a single logical "dimension" to fit GAP APIs
        self.dimensions = [RISK_DIMENSION_NAME]

    # ------------------------------------------------------------------    
    async def _predict_async(self, question: str, context: str) -> Tuple[float, Tuple[float, float], str]:
        """
        Returns (risk, (low, high), domain)
        """
        # Prefer service (it has domain + MemCube thresholds)
        if self._service:
            # Optional domain override
            if self.domain_override:
                domain = self.domain_override
            else:
                domain = await self._service.guess_domain(question or "")
            risk, (low, high) = await self._service.predict_risk(question or "", context or "", domain=domain)
            return float(risk), (float(low), float(high)), str(domain)

        # Local predictor path (model-layer)
        domain = self.domain_override
        if not domain:
            # Fallback local heuristic for domain resolution
            domain = await DomainCalibratedRiskPredictor._fallback_domain_resolver(question or "")  # type: ignore[attr-defined]

        risk, (low, high) = await self._predictor.predict_risk(question or "", context or "")  # type: ignore[union-attr]
        return float(risk), (float(low), float(high)), str(domain)

    def _predict_blocking(self, question: str, context: str) -> Tuple[float, Tuple[float, float], str]:
        """
        Best-effort sync wrapper for environments that call scorers synchronously.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We are already in an event loop: run task and wait
            fut = asyncio.run_coroutine_threadsafe(self._predict_async(question, context), loop)
            return fut.result()
        else:
            return asyncio.run(self._predict_async(question, context))

    # ------------------------------------------------------------------
    def _score_core(self, context: dict, scorable: Scorable, dimensions: List[str]) -> ScoreBundle:
        """
        Synchronous entry point (BaseScorer calls this). We do a best-effort
        sync call to the async predictor so existing pipelines work unchanged.
        """
        goal = context.get(GOAL, {})
        goal_text = goal.get(GOAL_TEXT, "") or ""
        response_text = scorable.text or ""

        try:
            risk, (low, high), domain = self._predict_blocking(goal_text, response_text)
        except Exception as e:
            self.logger.log("RiskScorerPredictError", {"error": str(e)})
            # Fail-soft: neutral risk with defaults
            risk, (low, high), domain = 0.0, (self.default_low, self.default_high), (self.domain_override or "general")

        band = "low" if risk < low else ("medium" if risk < high else "high")

        # Attributes (kept flat + vector-friendly)
        attrs: Dict[str, Any] = {
            "risk.score01": float(risk),
            "risk.low_threshold": float(low),
            "risk.high_threshold": float(high),
            "risk.band": band,
            "risk.domain": domain,
        }

        # Optional add-ons if service exposes them
        # (topological or coverage signals the downstream visuals like)
        try:
            if self._service:
                extra = awaitable_getattr(self._service, "last_topology")  # typed helper; returns dict or None
                if extra:
                    for k, v in extra.items():
                        attrs[f"risk.topo.{k}"] = float(v) if isinstance(v, (int, float)) else v
        except Exception:
            pass

        # Build aligned vector for VPM/PHOS (keep it small but informative)
        vec: Dict[str, float] = {
            "risk.score01": attrs["risk.score01"],
            "risk.low_threshold": attrs["risk.low_threshold"],
            "risk.high_threshold": attrs["risk.high_threshold"],
            # band one-hot for topology
            "risk.band.low": 1.0 if band == "low" else 0.0,
            "risk.band.medium": 1.0 if band == "medium" else 0.0,
            "risk.band.high": 1.0 if band == "high" else 0.0,
        }
        # Add minimal SCM alignment hooks (useful for dashboards)
        vec["scm.uncertainty01"] = float(risk)            # treat risk as uncertainty proxy
        vec["scm.aggregate01"]   = 1.0 - float(risk)      # “quality-ish” signal (optional)

        columns = list(vec.keys())
        values = [vec[c] for c in columns]
        attrs.update({"vector": vec, "columns": columns, "values": values})

        result = ScoreResult(
            dimension=RISK_DIMENSION_NAME,
            score=float(risk),
            source=self.model_type,
            rationale=f"risk={risk:.3f} band={band} [{domain}]",
            weight=1.0,
            attributes=attrs,
        )
        return ScoreBundle(results={RISK_DIMENSION_NAME: result})

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"<RiskScorer(model_type={self.model_type}, svc={bool(self._service)}, bundle={bool(self._predictor)})>"


# ------------------------
# Small async util
# ------------------------
def _isawaitable(x) -> bool:
    try:
        import inspect
        return inspect.isawaitable(x)
    except Exception:
        return False

def awaitable_getattr(obj, name: str):
    """
    If obj has attribute 'name' and it's awaitable, await it (best-effort).
    Otherwise return value or None.
    """
    try:
        val = getattr(obj, name, None)
        if _isawaitable(val):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(val, loop)
                return fut.result()
            else:
                return asyncio.run(val)
        return val
    except Exception:
        return None
