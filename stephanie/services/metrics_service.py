# stephanie/services/metrics_service.py
from __future__ import annotations
import asyncio, math, time, json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from stephanie.services.service_protocol import Service
from stephanie.scoring.metrics.registry import build_providers
from stephanie.scoring.metrics.providers.base import MetricProvider, MetricVector

@dataclass
class MetricsPost:
    zscore: Dict[str, Tuple[float, float]] | None = None
    clamp: Tuple[float, float] | None = None
    drop_zeros: bool = True

class MetricsService(Service):
    """
    Production-ready metrics builder:
      - Lifecycle (initialize/health/shutdown) per Service protocol
      - Concurrency limiting + timeouts
      - Bus wiring: subscribes to metrics requests, publishes results
      - Re-usable direct API: build(goal=..., text=..., context=...)
    """
    def __init__(self, cfg, memory, logger, container=None):
        self.cfg, self.memory, self.logger, self.container = cfg or {}, memory, logger, container

        # subjects (override in cfg.metrics.subjects)
        ms = (self.cfg.get("metrics", {}) or {}).get("subjects", {}) or {}
        self.subj_request: str = ms.get("request", "arena.metrics.request")
        self.subj_ready:   str = ms.get("ready",   "arena.metrics.ready")

        # runtime limits
        mcfg = (self.cfg.get("metrics") or {})
        self.max_concurrent: int = int(mcfg.get("max_concurrent", 16))
        self.timeout_sec:    float = float(mcfg.get("timeout_sec", 12.0))

        self._sem = asyncio.Semaphore(self.max_concurrent)
        self._initialized = False
        self._stopped = False
        self._providers: List[MetricProvider] = []
        self._post = MetricsPost(
            zscore = mcfg.get("zscore_stats"),
            clamp  = tuple(mcfg["min_max"]) if isinstance(mcfg.get("min_max"), (list,tuple)) and len(mcfg["min_max"])==2 else None,
            drop_zeros = bool(mcfg.get("drop_all_zero_features", True)),
        )

        # stats
        self._stats = {
            "processed": 0,
            "failed": 0,
            "in_flight": 0,
            "last_error": None,
            "uptime_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # -------- Service protocol

    async def initialize(self, **kwargs) -> None:
        if self._initialized:  # idempotent
            return
        # Build providers once
        self._providers = build_providers(self.cfg, self.container, self.memory, self.logger)

        # Attach bus if container/memory configured it already
        bus = getattr(self.memory, "bus", None)
        if bus:
            # subscribe to request subject
            async def _handler(msg: Dict[str, Any]):
                # msg is already a dict; support raw bytes fallback
                if isinstance(msg, (bytes, bytearray)):
                    try:
                        payload = json.loads(msg.decode("utf-8"))
                    except Exception:
                        payload = {}
                else:
                    payload = msg
                await self._on_request(payload)

            await bus.subscribe(self.subj_request, _handler)

        self._initialized = True
        lg = getattr(self.logger, "log", None)
        if callable(lg):
            lg("MetricsServiceInitialized", {
                "providers": [getattr(p, "name", type(p).__name__) for p in self._providers],
                "subjects": {"request": self.subj_request, "ready": self.subj_ready},
                "max_concurrent": self.max_concurrent,
                "timeout_sec": self.timeout_sec,
            })

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized and not self._stopped else "uninitialized" if not self._initialized else "stopped",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {**self._stats, "max_concurrent": self.max_concurrent, "timeout_sec": self.timeout_sec},
        }

    def shutdown(self) -> None:
        self._stopped = True
        self._providers = []
        lg = getattr(self.logger, "log", None)
        if callable(lg):
            lg("MetricsServiceShutdown", {})

    @property
    def name(self) -> str:
        return "metrics-service-v1"

    # -------- Bus handlers

    async def _on_request(self, envelope: Dict[str, Any]) -> None:
        """
        Handle a single metrics request event:
            {
              "run_id": "...", "node_id": "...",
              "goal_text": "...", "prompt_text": "...",
              ... any extras ...
            }
        Publishes:
            subject=self.subj_ready with payload:
            {
              "run_id": "...", "node_id": "...",
              "vector": {"names":[...],"values":[...]},
              "meta": {"dim": N, "providers":[...]},
              "ts": <unix>
            }
        """
        if self._stopped:
            return
        goal = (envelope.get("goal_text") or "").strip()
        text = (envelope.get("prompt_text") or "").strip()
        if not text:
            # nothing to do
            return

        await self._run_job(envelope, goal, text)

    async def _run_job(self, envelope: Dict[str, Any], goal: str, text: str) -> None:
        bus = getattr(self.memory, "bus", None)
        async with self._sem:
            self._stats["in_flight"] += 1
            try:
                result = await asyncio.wait_for(self.build(goal=goal, text=text, context=envelope), timeout=self.timeout_sec)
                payload = {
                    "run_id": envelope.get("run_id"),
                    "node_id": envelope.get("node_id"),
                    "vector": result["vector"],
                    "meta": result["meta"],
                    "ts": time.time(),
                }
                if bus:
                    await bus.publish(self.subj_ready, payload)
                self._stats["processed"] += 1
            except Exception as e:
                self._stats["failed"] += 1
                self._stats["last_error"] = str(e)
                lg = getattr(self.logger, "log", None)
                if callable(lg):
                    lg("MetricsJobError", {"error": str(e), "envelope": {k: envelope.get(k) for k in ("run_id","node_id")}})
            finally:
                self._stats["in_flight"] = max(0, self._stats["in_flight"] - 1)

    # -------- Public API

    async def build(self, *, goal: str, text: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Direct API that computes a metrics vector for (goal, text).
        Returns: { "vector": {"names":[...], "values":[...]}, "meta": {...} }
        """
        ctx = context or {}

        # compute providers concurrently
        tasks = [asyncio.create_task(p.compute(goal=goal, text=text, context=ctx)) for p in self._providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        names: List[str] = []
        values: List[float] = []

        for p, r in zip(self._providers, results):
            if isinstance(r, Exception):
                self._log_err(p, r)
                continue
            # r is MetricVector
            names.extend(r.names)
            values.extend(r.values)

        # normalize/filter
        names, values = self._postprocess(names, values)

        return {
            "vector": {"names": names, "values": values},
            "meta": {
                "providers": [{"name": getattr(p, "name", type(p).__name__), "version": getattr(p, "version", "0.0.0")} for p in self._providers],
                "dim": len(names),
            },
        }

    # -------- Post-processing

    def _postprocess(self, names: List[str], values: List[float]) -> Tuple[List[str], List[float]]:
        # z-score
        if self._post.zscore:
            nn, vv = [], []
            for n, v in zip(names, values):
                mu, sd = self._post.zscore.get(n, (0.0, 1.0))
                sd = max(sd, 1e-8)
                z = (v - mu) / sd if sd > 0 else 0.0
                if math.isnan(z): z = 0.0
                nn.append(n); vv.append(z)
            names, values = nn, vv

        # clamp-> [0,1]
        if self._post.clamp:
            lo, hi = self._post.clamp
            span = hi - lo if hi != lo else 1.0
            values = [max(min(v, hi), lo) for v in values]
            values = [(v - lo) / span for v in values]

        # drop zeros
        if self._post.drop_zeros:
            keep = [(n, v) for n, v in zip(names, values) if v != 0.0]
            if keep:
                names, values = list(zip(*keep))
                names, values = list(names), list(values)
            else:
                names, values = [], []

        return names, values

    def _log_err(self, provider: MetricProvider, err: Exception):
        if hasattr(self.logger, "log") and callable(getattr(self.logger, "log")):
            self.logger.log("MetricsProviderError", {"provider": getattr(provider, "name", type(provider).__name__), "error": str(err)})
