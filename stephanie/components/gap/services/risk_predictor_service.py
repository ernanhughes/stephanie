from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np

from stephanie.services.service_protocol import Service
from stephanie.scoring.model.risk_predictor import DomainCalibratedRiskPredictor

# Optional, if MemCube client is available in your env
try:
    from stephanie.memcube import MemCubeClient  # type: ignore
except Exception:
    MemCubeClient = None  # graceful fallback


@dataclass
class RiskServiceConfig:
    bundle_path: str = "./models/risk/bundle.joblib"
    # If you already store per-domain calibration in MemCube, leave this empty.
    # Otherwise we will use these domains for cold-start thresholds.
    default_domains: tuple[str, ...] = ("science", "history", "geography", "tech")
    # Cache TTL (seconds) for domain thresholds pulled from MemCube
    calib_ttl_s: int = 3600
    # Hard fallback thresholds if neither MemCube nor model returns thresholds
    fallback_low: float = 0.20
    fallback_high: float = 0.60
    # SHAP explanations can be expensive; keep opt-in
    enable_explanations: bool = False


class RiskPredictorService(Service):
    """
    Wraps DomainCalibratedRiskPredictor in a Service-compatible facade.

    Public methods for consumers:
        - await predict_risk(question: str, context: str, domain_hint: Optional[str]) -> tuple[float, tuple[float,float]]
        - await get_domain_thresholds(domain: str) -> tuple[float,float]
        - await explain_risk(question: str, context: str) -> Optional[bytes]  (PNG bytes if enabled)
    """

    def __init__(self, cfg: Optional[RiskServiceConfig] = None, logger: Optional[logging.Logger] = None):
        self._cfg = cfg or RiskServiceConfig()
        self._logger = logger or logging.getLogger(self.name)

        self._predictor: Optional[DomainCalibratedRiskPredictor] = None
        self._mem: Optional[Any] = None  # MemCubeClient if available
        self._up: bool = False
        self._req_count: int = 0
        self._err_count: int = 0
        self._lat_ema_ms: float = 0.0

        # Domain threshold cache {domain: (low, high, expires_at)}
        self._th_cache: Dict[str, Tuple[float, float, float]] = {}
        self._cache_lock = asyncio.Lock()

        # For event bus subscription management
        self._bus = None
        self._bus_ready = asyncio.Event()

    # ---------- Service protocol ----------
    @property
    def name(self) -> str:
        return "policy-risk-v2"

    def initialize(self, **kwargs) -> None:
        # Merge external init kwargs into our config (non-destructive)
        cfg_kw = kwargs.get("config") or {}
        if cfg_kw:
            for k, v in cfg_kw.items():
                if hasattr(self._cfg, k):
                    setattr(self._cfg, k, v)

        # Dedicated logger if provided
        lg = kwargs.get("logger")
        if lg is not None:
            self._logger = lg

        # Instantiate predictor
        self._predictor = DomainCalibratedRiskPredictor(
            bundle_path=self._cfg.bundle_path,
            default_domains=self._cfg.default_domains,
        )

        # Optional MemCube (used for calibration fetch/persist)
        if MemCubeClient is not None:
            try:
                self._mem = MemCubeClient()
            except Exception:
                self._mem = None

        self._up = True
        self._logger.info(
            "RiskPredictorService initialized",
            extra={
                "service": self.name,
                "bundle_path": self._cfg.bundle_path,
                "domains": list(self._cfg.default_domains),
            },
        )

        # If a bus was injected before initialize(), finish subscriptions
        if self._bus is not None:
            asyncio.create_task(self._subscribe_bus())

    def set_bus(self, bus: Any) -> None:
        """Attach event bus and subscribe to calibration updates."""
        self._bus = bus
        super().set_bus(bus)  # logs bus attach
        # If we're already initialized, subscribe now
        if self._up:
            asyncio.create_task(self._subscribe_bus())

    async def _subscribe_bus(self):
        """Subscribe to calibration updates to hot-reload thresholds cache."""
        if not self._bus:
            return
        try:
            await self.subscribe(
                subject="calibration.risk.updated",
                handler=self._on_calibration_updated,
                queue_group="risk-calibration",
            )
            self._bus_ready.set()
            self._logger.info("RiskPredictorService subscribed to calibration.risk.updated")
        except Exception as e:
            self._logger.error(f"Failed subscribing to bus: {e}")

    async def _on_calibration_updated(self, payload: Dict[str, Any]):
        """
        Payload example:
            {"domain": "geography", "low_threshold": 0.15, "high_threshold": 0.55, "ts": "..."}
        """
        domain = str(payload.get("domain") or "").strip()
        if not domain:
            return
        low = payload.get("low_threshold")
        high = payload.get("high_threshold")
        if low is None or high is None:
            return

        async with self._cache_lock:
            self._th_cache[domain] = (float(low), float(high), time.time() + self._cfg.calib_ttl_s)
        self._logger.info(
            "Risk thresholds hot-reloaded",
            extra={"domain": domain, "low": low, "high": high},
        )

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
                "memcube": "available" if self._mem is not None else "unavailable",
                "bus": "connected" if self._bus is not None else "none",
            },
        }

    def shutdown(self) -> None:
        self._up = False
        self._predictor = None
        self._th_cache.clear()
        self._logger.info("RiskPredictorService shutdown complete")

    # ---------- Public API for consumers ----------
    async def predict_risk(
        self,
        question: str,
        context: str,
        *,
        domain_hint: Optional[str] = None,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Return risk and (low, high) thresholds. Thresholds are fetched in order of preference:
          1) Model's own per-domain calibration via DomainCalibratedRiskPredictor
          2) Cached thresholds from MemCube (hot cache)
          3) Cold fetch from MemCube (populate cache)
          4) Hard-coded fallbacks in config
        """
        if not self._predictor:
            raise RuntimeError("RiskPredictorService not initialized")

        t0 = time.perf_counter()
        try:
            # Use predictor (which already performs its own calibration lookup)
            risk, (low, high) = await self._predictor.predict_risk(question, context)

            # If the predictor can't provide thresholds, consult MemCube/cache
            if low is None or high is None:
                domain = domain_hint or await self._guess_domain_safe(question)
                low, high = await self.get_domain_thresholds(domain)

            self._record_latency(t0)
            self._req_count += 1
            return float(risk), (float(low), float(high))
        except Exception as e:
            self._err_count += 1
            self._logger.error(f"predict_risk failed: {e}")
            # Always provide some thresholds
            return 0.0, (self._cfg.fallback_low, self._cfg.fallback_high)

    async def get_domain_thresholds(self, domain: Optional[str]) -> Tuple[float, float]:
        """
        Best-effort domain thresholds with TTL cache and MemCube fallback.
        """
        # Normalize domain
        d = (domain or "").strip().lower() or "general"

        # 1) Hot cache
        now = time.time()
        async with self._cache_lock:
            cached = self._th_cache.get(d)
            if cached and cached[2] > now:
                return cached[0], cached[1]

        # 2) MemCube lookup
        low, high = None, None
        if self._mem:
            try:
                rec = await self._mem.query_calibration(
                    "risk",
                    filters={"domain": d},
                    sort=[("created_at", "DESC")],
                    limit=1,
                )
                if rec:
                    low = float(rec.get("low_threshold"))
                    high = float(rec.get("high_threshold"))
            except Exception:
                pass

        # 3) Defaults
        if low is None or high is None:
            low = self._cfg.fallback_low
            high = self._cfg.fallback_high

        # Cache it
        async with self._cache_lock:
            self._th_cache[d] = (low, high, now + self._cfg.calib_ttl_s)

        return low, high

    async def explain_risk(self, question: str, context: str) -> Optional[bytes]:
        """
        Optional SHAP explanation as PNG bytes.
        Returns None if explanations are disabled or unavailable.
        """
        if not self._cfg.enable_explanations:
            return None
        if not self._predictor:
            return None

        try:
            # Many SHAP toolings require a sync call; wrap in thread if needed
            # We assume predictor exposes `explain_to_png(question, context) -> bytes`
            if hasattr(self._predictor, "explain_to_png"):
                loop = asyncio.get_event_loop()
                png_bytes: bytes = await loop.run_in_executor(
                    None, lambda: self._predictor.explain_to_png(question, context)
                )
                return png_bytes
        except Exception as e:
            self._logger.warning(f"explain_risk failed: {e}")
        return None

    # ---------- internal helpers ----------
    def _record_latency(self, t0: float) -> None:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # Simple EMA with alpha=0.05
        a = 0.05
        self._lat_ema_ms = (1 - a) * self._lat_ema_ms + a * dt_ms if self._lat_ema_ms > 0 else dt_ms

    async def _guess_domain_safe(self, question: str) -> str:
        """Fallback domain guess if MemCube is not available via predictor."""
        if self._predictor and hasattr(self._predictor, "memcube"):
            try:
                d = await self._predictor.memcube.guess_domain(question)  # type: ignore
                if d:
                    return str(d)
            except Exception:
                pass
        # naive heuristic fallback
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
