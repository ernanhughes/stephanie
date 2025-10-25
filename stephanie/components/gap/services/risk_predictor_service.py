# stephanie/components/gap/services/risk_predictor_service.py
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from stephanie.services.service_protocol import Service
from stephanie.scoring.model.risk_predictor import DomainCalibratedRiskPredictor

from stephanie.memcube.memcube_client import MemCubeClient
from stephanie.memory.memcube_store import MemcubeStore
from stephanie.analysis.scorable_classifier import ScorableClassifier


@dataclass
class RiskServiceConfig:
    bundle_path: str = "./models/risk/bundle.joblib"
    default_domains: tuple[str, ...] = ("science", "history", "geography", "tech", "general")
    calib_ttl_s: int = 3600
    fallback_low: float = 0.20
    fallback_high: float = 0.60
    enable_explanations: bool = False
    # Optional direct injection
    domain_seed_config_path: str = "config/domain/seeds.yaml"
    memcube: Any = None


def _coerce_cfg(cfg: Optional[Union[RiskServiceConfig, dict, Any]]) -> RiskServiceConfig:
    """
    Accept RiskServiceConfig, dict, or a large config object (e.g., GapConfig).
    Try to extract a nested 'risk' or 'risk_predictor' block; otherwise use defaults.
    """
    if cfg is None:
        return RiskServiceConfig()

    if isinstance(cfg, RiskServiceConfig):
        return cfg

    if isinstance(cfg, dict):
        # Allow nested shapes like {"risk": {...}} or {"risk_predictor": {...}}
        sub = cfg.get("risk") or cfg.get("risk_predictor") or cfg
        return RiskServiceConfig(**{k: v for k, v in sub.items() if k in RiskServiceConfig().__dict__})

    # Fallback for big config objects (Hydra/OmegaConf/GapConfig)
    # Try common attribute names
    for attr in ("risk", "risk_predictor", "policy_risk", "eg"):
        if hasattr(cfg, attr):
            block = getattr(cfg, attr)
            if isinstance(block, dict):
                return _coerce_cfg(block)
            # Support dataclass-like with __dict__
            if hasattr(block, "__dict__"):
                return _coerce_cfg(block.__dict__)

    # Last resort: defaults
    return RiskServiceConfig()


class RiskPredictorService(Service):
    def __init__(self, cfg: Optional[Union[RiskServiceConfig, dict, Any]], memory, logger: Optional[logging.Logger] = None):
        self.cfg = _coerce_cfg(cfg)
        self.memory = memory
        self.logger = logger or logging.getLogger("policy-risk-v2")

        self._predictor: Optional[DomainCalibratedRiskPredictor] = None
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

    @property
    def name(self) -> str:
        return "policy-risk-v2"

    def initialize(self, **kwargs) -> None:
        # Merge/override config if present
        incoming = kwargs.get("config")
        if incoming is not None:
            self.cfg = _coerce_cfg(incoming)

        lg = kwargs.get("logger")
        if lg is not None:
            self.logger = lg

        # Grab a memcube client if available:
        # 1) explicit in config
        # 2) memory.memcube (your store-backed client)
        self.memcubes = self.memory.memcubes

        # Instantiate predictor (pass memcube if present)
        self._predictor = DomainCalibratedRiskPredictor(
            bundle_path=self.cfg.bundle_path,
            default_domains=list(self.cfg.default_domains),
            memcube=self.memcubes,
            domain_classifier=self.domain_classifier,
        )

        self._up = True
        self.logger.info(
            "RiskPredictorService initialized",
            extra={
                "service": self.name,
                "bundle_path": self.cfg.bundle_path,
                "domains": list(self.cfg.default_domains),
                "memcube": bool(self.memcubes),
            },
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
                "bundle": "loaded" if self._predictor is not None else "missing",
                "memcube": "available" if self.memcubes is not None else "unavailable",
                "bus": "connected" if self.bus is not None else "none",
            },
        }

    def shutdown(self) -> None:
        self._up = False
        self._predictor = None
        self._th_cache.clear()
        self.logger.info("RiskPredictorService shutdown complete")

    # --- public API ---
    async def predict_risk(self, question: str, context: str, *, domain_hint: Optional[str] = None) -> Tuple[float, Tuple[float, float]]:
        if not self._predictor:
            raise RuntimeError("RiskPredictorService not initialized")

        t0 = time.perf_counter()
        try:
            risk, (low, high) = await self._predictor.predict_risk(question, context)
            if low is None or high is None:
                domain = (domain_hint or await self._guess_domain_safe(question)).lower()
                low, high = await self.get_domain_thresholds(domain)
            self._record_latency(t0)
            self._req_count += 1
            return float(risk), (float(low), float(high))
        except Exception as e:
            self._err_count += 1
            self.logger.error(f"predict_risk failed: {e}")
            return 0.0, (self.cfg.fallback_low, self.cfg.fallback_high)

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
                rec = await self.memcubes.query_calibration("risk", filters={"domain": d}, sort=[("created_at", "DESC")], limit=1)
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

    async def explain_risk(self, question: str, context: str) -> Optional[bytes]:
        if not self.cfg.enable_explanations or not self._predictor:
            return None
        try:
            if hasattr(self._predictor, "explain_to_png"):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: self._predictor.explain_to_png(question, context))
        except Exception as e:
            self.logger.warning(f"explain_risk failed: {e}")
        return None

    def reload_bundle(self) -> bool:
        """Called by your trainer agent after writing a new bundle.joblib."""
        try:
            self._predictor = DomainCalibratedRiskPredictor(
                bundle_path=self.cfg.bundle_path,
                default_domains=list(self.cfg.default_domains),
                memcube=self.memcubes,
            )
            self.logger.info("RiskPredictorService bundle reloaded", extra={"bundle_path": self.cfg.bundle_path})
            return True
        except Exception as e:
            self.logger.error(f"reload_bundle failed: {e}")
            return False

    # --- internals ---
    def _record_latency(self, t0: float) -> None:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        a = 0.05
        self._lat_ema_ms = (1 - a) * self._lat_ema_ms + a * dt_ms if self._lat_ema_ms > 0 else dt_ms

    async def _guess_domain_safe(self, question: str) -> str:
        # Prefer predictor’s memcube if present
        try:
            if self._predictor and getattr(self._predictor, "memcube", None):
                d = await self._predictor.memcube.guess_domain(question)  # type: ignore[attr-defined]
                if d:
                    return str(d)
        except Exception:
            pass
        # fallback heuristic
        ql = (question or "").lower()
        if any(k in ql for k in ("who", "born", "died", "year", "reign", "empire")):
            return "history"
        if any(k in ql for k in ("river", "border", "capital", "country", "latitude", "longitude")):
            return "geography"
        if any(k in ql for k in ("algorithm", "api", "software", "cpu", "gpu")):
            return "tech"
        if any(k in ql for k in ("cell", "atom", "physics", "chemical", "species")):
            return "science"
        return "general"
