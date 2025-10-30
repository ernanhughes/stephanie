# stephanie/components/risk/services/risk_predictor_service.py
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.memory.memcube_store import MemcubeStore
from stephanie.scoring.scorable import Scorable
from stephanie.components.risk.features.entity_domain_features import compute_entity_metrics, risk_from_features
from stephanie.services.service_protocol import Service

_logger = logging.getLogger(__name__)

@dataclass
class RiskServiceConfig:
    bundle_path: str = "./models/risk/bundle.joblib"
    default_domains: tuple[str, ...] = (
        "programming","apis","systems","devops","cloud","security",
        "databases","data_science","ml","nlp","vision",
        "math","research","ethics","general"
    )
    calib_ttl_s: int = 3600
    fallback_low: float = 0.20
    fallback_high: float = 0.60
    enable_explanations: bool = False
    domain_seed_config_path: str = "config/domain/seeds_tech.yaml"

class RiskPredictorService(Service):
    """
    STRICT: never guess domains.
    - For scorables: domain must come from scorable.domains; if absent, we run the classifier explicitly.
    - For raw text: use predict_risk_strict(question, context, domain=...).
    """
    def __init__(self, cfg: RiskServiceConfig, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        self.memcubes: MemcubeStore = memory.memcubes
        self._up: bool = False
        self._req_count: int = 0
        self._err_count: int = 0
        self._lat_ema_ms: float = 0.0
        self._th_cache: Dict[str, Tuple[float, float, float]] = {}
        self._cache_lock = asyncio.Lock()

        self.domain_classifier = ScorableClassifier(
            memory,
            logger,
            self.cfg.domain_seed_config_path or "config/domain/seeds.yaml",
        )
        self.bus = memory.bus
        self._bus_ready = asyncio.Event()

    @property
    def name(self) -> str:
        return "risk-predictor-service"

    def initialize(self, **kwargs) -> None:
        if (incoming := kwargs.get("config")) is not None:
            self.cfg = RiskServiceConfig(**incoming)
        if (lg := kwargs.get("logger")) is not None:
            self.logger = lg

        self._up = True
        self.logger.info(
            "RiskPredictorService initialized",
            extra={"service": self.name, "domains": list(self.cfg.default_domains), "memcube": bool(self.memcubes)},
        )
        if self.bus is not None:
            asyncio.create_task(self._subscribe_bus())

    def set_bus(self, bus: Any) -> None:
        self.bus = bus
        super().set_bus(bus)
        if self._up:
            asyncio.create_task(self._subscribe_bus())

    async def _subscribe_bus(self):
        if not self.bus:
            return
        try:
            await self.subscribe(
                subject="calibration.risk.updated",
                handler=self._on_calibration_updated,
                queue_group="risk-calibration",
            )
            self._bus_ready.set()
            self.logger.info("Subscribed to calibration.risk.updated")
        except Exception as e:
            self.logger.error(f"Bus subscribe failed: {e}")

    async def _on_calibration_updated(self, payload: Dict[str, Any]):
        domain = str(payload.get("domain") or "").strip().lower()
        if not domain:
            return
        low = payload.get("low_threshold")
        high = payload.get("high_threshold")
        if low is None or high is None:
            return
        async with self._cache_lock:
            self._th_cache[domain] = (float(low), float(high), time.time() + self.cfg.calib_ttl_s)
        self.logger.info("Risk thresholds hot-reloaded", extra={"domain": domain, "low": low, "high": high})

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._up else "unhealthy",
            "metrics": {
                "requests": self._req_count,
                "errors": self._err_count,
                "latency_ema_ms": round(self._lat_ema_ms, 2),
                "domains_cached": len(self._th_cache),
            },
            "dependencies": {
                "memcube": "available" if self.memcubes is not None else "unavailable",
                "bus": "connected" if self.bus is not None else "none",
            },
        }

    def shutdown(self) -> None:
        self._up = False
        self._th_cache.clear()
        self.logger.info("RiskPredictorService shutdown complete")

    # ---------------- PUBLIC: strict API ----------------

    async def predict_for_scorable(
        self,
        scorable: Scorable,
        reply_text: str,
        *,
        evidence_text: str = "",
        min_domain_conf: float = 0.0,
    ) -> Tuple[float, Tuple[float, float], Dict[str, Any]]:
        """
        Strict path: domain must be present on scorable.domains; if empty, we run the classifier (deterministic).
        No memcube.guess_domain anywhere.
        """
        t0 = time.perf_counter()
        try:
            # 1) Domain: from scorable, or classify deterministically if missing
            domains = scorable.domains or []
            if not domains:
                # Deterministic classification (not a heuristic "guess")
                cls = self.domain_classifier.classify(scorable.text, top_k=1, min_value=0.0)
                domains = [{"domain": d, "score": float(s), "source": "seed"} for d, s in (cls or [])]
            domain_used = Scorable.pick_primary_domain(domains, min_conf=min_domain_conf)

            # 2) Entity metrics (NER comes from scorable)
            ent_feats = {} # await compute_entity_metrics(self.memory, domain_used, scorable.ner or [])

            # 3) Coverage / OOV / numeric checks (stubs → wire your own)
            coverage_overlap = await self._coverage_overlap(scorable.text, evidence_text or "")
            oov_rate = self._domain_oov_rate(domain_used, reply_text or "")
            numeric_incons = self._numeric_inconsistency(domain_used, scorable.text or "", reply_text or "")

            feats = {
                **ent_feats,
                "coverage_overlap": coverage_overlap,
                "oov_rate": oov_rate,
                "numeric_inconsistency": numeric_incons,
            }

            # 4) Deterministic risk
            risk = risk_from_features(domain_used, feats)

            # 5) Domain-calibrated thresholds (MemCube)
            low, high = await self.get_domain_thresholds(domain_used)

            # 6) Bookkeeping
            self._record_latency(t0)
            self._req_count += 1
            meta = {"domain": domain_used, "features": feats}
            return float(risk), (float(low), float(high)), meta

        except Exception as e:
            self._err_count += 1
            self.logger.error(f"predict_for_scorable failed: {e}")
            return 0.0, (self.cfg.fallback_low, self.cfg.fallback_high), {"error": str(e)}

    async def predict_risk_strict(
        self,
        question: str,
        context: str,
        *,
        domain: str,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        For callers without a Scorable but who DO know the domain.
        Required: domain (no guessing).
        """
        if not domain:
            raise ValueError("predict_risk_strict requires a domain.")
        # Minimal deterministic baseline without entities:
        coverage_overlap = await self._coverage_overlap(question, context)
        feats = {
            "entity_support_ratio": 1.0,   # unknown → default safe
            "entity_contradiction_ratio": 0.0,
            "entity_unresolved_ratio": 0.0,
            "entity_ambiguity": 0.0,
            "coverage_overlap": coverage_overlap,
            "oov_rate": 0.0,
            "numeric_inconsistency": 0.0,
        }
        risk = risk_from_features(domain, feats)
        low, high = await self.get_domain_thresholds(domain)
        return float(risk), (float(low), float(high))

    # ---------------- THRESHOLDS ----------------

    async def get_domain_thresholds(self, domain: Optional[str]) -> Tuple[float, float]:
        d = (domain or "").strip().lower() or "general"
        now = time.time()
        async with self._cache_lock:
            cached = self._th_cache.get(d)
            if cached and cached[2] > now:
                return cached[0], cached[1]

        low = high = None
        if self.memcubes:
            try:
                rec = await self.memcubes.query_calibration(
                    "risk", filters={"domain": d}, sort=[("created_at", "DESC")], limit=1
                )
                if rec:
                    low = float(rec.get("low_threshold"))
                    high = float(rec.get("high_threshold"))
            except Exception as e:
                self.logger.warning(f"MemCube calibration fetch failed: {e}")

        if low is None or high is None:
            low = self.cfg.fallback_low
            high = self.cfg.fallback_high

        async with self._cache_lock:
            self._th_cache[d] = (low, high, now + self.cfg.calib_ttl_s)
        return low, high

    # ---------------- INTERNALS ----------------

    def _record_latency(self, t0: float) -> None:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        a = 0.05
        self._lat_ema_ms = (1 - a) * self._lat_ema_ms + a * dt_ms if self._lat_ema_ms > 0 else dt_ms

    async def _coverage_overlap(self, score_text: str, evidence_text: str) -> float:
        if not score_text or not evidence_text:
            return 0.0
        a = set(score_text.lower().split())
        b = set(evidence_text.lower().split())
        return len(a & b) / max(1, len(a))

    def _domain_oov_rate(self, domain: str, text: str) -> float:
        tokens = [t for t in (text or "").split() if t.isalpha()]
        if not tokens:
            return 0.0
        known = sum(1 for t in tokens if len(t) <= 30)  # replace with your lexicon
        return 1.0 - (known / len(tokens))

    def _numeric_inconsistency(self, domain: str, goal: str, reply: str) -> float:
        # Hook up your unit/version/date checking here (0..1)
        return 0.0
