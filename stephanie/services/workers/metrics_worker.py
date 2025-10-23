# stephanie/services/workers/metrics_worker.py
from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, Optional

from stephanie.scoring.scorable import Scorable
from stephanie.services.scoring_service import ScoringService

_logger = logging.getLogger(__name__)


class MetricsWorkerInline:
    """Simplified in-process scorer (no bus)."""
    def __init__(self, scoring: ScoringService, scorers: list[str], dimensions: list[str], persist=False):
        self.scoring = scoring
        self.scorers = scorers
        self.dimensions = dimensions
        self.persist = persist

    async def score(self, scorable: Scorable, goal_text: str, run_id: str) -> dict:
        ctx = {"goal": {"goal_text": goal_text}, "pipeline_run_id": run_id}
        vector, results = {}, {}
        model_aliases = []
        for name in self.scorers:
            bundle = (self.scoring.score_and_persist if self.persist else self.scoring.score)(
                scorer_name=name, scorable=scorable, context=ctx, dimensions=self.dimensions
            )
            model_alias = self.scoring.get_model_name(name)
            model_aliases.append(model_alias)
            agg = float(bundle.aggregate())
            per = {d: float(sr.score) for d, sr in bundle.results.items()}
            results[name] = {"aggregate": agg, "per_dimension": per}

            flat = bundle.flatten(include_scores=True, include_attributes=True, numeric_only=True)
            for k, v in flat.items():
                vector[f"{model_alias}.{k}"] = float(v)
            vector[f"{model_alias}.aggregate"] = agg


            # yield to event loop so progress can flush in long runs
            await asyncio.sleep(0)

        columns = sorted(vector.keys())
        values = [vector[c] for c in columns]
        return {"model_alias": model_aliases, "columns": columns, "values": values, "vector": vector, "scores": results}




class MetricsWorker:
    """
    Consumes 'metrics.request', scores with configured scorers,
    and publishes 'metrics.ready'.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg       = cfg
        self.memory    = memory
        self.container = container
        self.logger    = logger
        mcfg = (cfg.get("metrics") or {})
        self.subj_req  = ((mcfg.get("subjects") or {}).get("request")
                          or "arena.metrics.request")
        self.subj_ready= ((mcfg.get("subjects") or {}).get("ready")
                          or "arena.metrics.ready")
        self.scorers   = list(mcfg.get("scorers") or ["sicql", "mrq", "ebt"])   # names
        self.dimensions = list(mcfg.get("dimensions") or ["alignment", "clarity", "implementability", "novelty", "relevance"])    
        
        self.persist   = bool(mcfg.get("persist", False))
        self.timeout_s = float(mcfg.get("timeout_s", 10))
        self.scorable_type = str(mcfg.get("scorable_type", "response"))
        self.max_conc  = int(mcfg.get("max_concurrency", 8))
        self.sem       = asyncio.Semaphore(self.max_conc)

        # get ScoringService from container
        self.scoring: ScoringService = self.container.get("scoring")

        # optional: simple de-dupe cache to avoid double-processing
        self._seen = set()  # {(run_id, node_id)}

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._health_task = None


    async def handle_job(self, envelope: Dict[str, Any]):
        try:
            run_id  = envelope.get("run_id")
            node_id = envelope.get("node_id")
            if not run_id or not node_id:
                # nothing to publish if key ids are missing
                return

            t0 = time.time()
            goal_text   = (envelope.get("goal_text") or "").strip()
            prompt_text = (envelope.get("prompt_text") or "").strip()
            if not goal_text or not prompt_text:
                out = {**envelope, "error": "missing_fields", "ts_completed": time.time()}
                await self.memory.bus.publish(subject=self.subj_ready, payload=out)
                return

            # goal-conditioned context for scorers
            ctx = {
                "goal": {"goal_text": goal_text},
                "pipeline_run_id": run_id,
            }

            scorable = Scorable(id=None, text=prompt_text, target_type=self.scorable_type)

            # build one flat vector across all scorers
            vector: Dict[str, float] = {}
            results: Dict[str, Any] = {}  # human-friendly summary per scorer

            for name in self.scorers:
                try:
                    bundle = (
                        self.scoring.score_and_persist(
                            scorer_name=name,
                            scorable=scorable,
                            context=ctx,
                            dimensions=self.dimensions,
                        )
                        if self.persist else
                        self.scoring.score(
                            scorer_name=name,
                            scorable=scorable,
                            context=ctx,
                            dimensions=self.dimensions,
                        )
                    )

                    agg = float(bundle.aggregate())
                    per = {d: float(sr.score) for d, sr in bundle.results.items()}
                    results[name] = {
                        "aggregate": agg,
                        "per_dimension": per,
                        "persisted": bool(self.persist),
                    }

                    flat = bundle.flatten(
                        include_scores=True,
                        include_weights=False,
                        include_sources=False,
                        include_rationales=False,
                        include_attributes=True,
                        include_meta=False,
                        numeric_only=True,
                        sep=".",
                        attr_prefix="attr",
                    )
                    for k, v in flat.items():
                        vector[f"{name}.{k}"] = float(v)
                    vector[f"{name}.aggregate"] = agg

                except Exception as e:
                    self.logger.log("MetricsWorkerScoreError", {"scorer": name, "error": str(e)})
                    results[name] = {"error": str(e)}

            # deterministic column order for the VPM (columns = metric names)
            columns = sorted(vector.keys())
            values = [vector[c] for c in columns]
            latency_ms = (time.time() - t0) * 1000.0

            # SINGLE publish, including the entire original envelope
            out = {
                **envelope,                      # include original envelope fields
                "scores": results,               # human-friendly per-scorer summary
                "metrics_columns": columns,      # stable x-axis
                "metrics_values": values,        # aligned numeric vector
                "metrics_vector": vector,        # full mapping (optional but handy)
                "latency_ms": latency_ms,
                "ts_completed": time.time(),
            }

            await self.memory.bus.publish(subject=self.subj_ready, payload=out)

        except Exception as e:
            self.logger.log("MetricsWorkerJobError", {
                "error": str(e),
                "trace": traceback.format_exc(),
                "envelope": envelope,
            })

    async def start(self):
        if self._running:
            return
        self._running = True
        # subscribe to bus
        await self.memory.bus.subscribe(self.subj_req, self.handle_job)
        self.logger.log("MetricsWorkerStarted", {"subject": self.subj_req})

    async def stop(self):
        self._running = False
        # if your bus supports unsubscribe:
        try:
            await self.memory.bus.unsubscribe(self.subj_req, self.handle_job)
        except Exception:
            pass
        self.logger.log("MetricsWorkerStopped", {})

    async def _monitor_bus_health(self):
        """Continuously monitor bus health with detailed logging"""
        while True:
            try:
                health = self.memory.bus.health_check()
                bus_type = health["bus_type"]
                status = health["status"]
                details = health["details"]
                
                # Format detailed health info
                if bus_type == "nats":
                    details_str = (
                        f"connected={details.get('connected', '?')}, "
                        f"closed={details.get('closed', '?')}, "
                        f"uptime={details.get('connection_uptime', 0):.1f}s, "
                        f"reconnects={details.get('reconnect_attempts', 0)}"
                    )
                elif bus_type == "inproc":
                    details_str = (
                        f"subscriptions={details.get('subscriptions', '?')}, "
                        f"mem={details.get('memory_usage', '?')}"
                    )
                else:
                    details_str = str(details)
                
                # Log with color-coded status
                if status:
                    _logger.debug("BUS HEALTH: 🟢 %s - %s | %s", bus_type, status, details_str)
                elif status == "disconnected":
                    _logger.warning("BUS HEALTH: 🔴 %s - %s | %s", bus_type, status, details_str)
                
                await asyncio.sleep(30)
            except Exception as e:
                _logger.error("BUS HEALTH CHECK FAILED: %s", str(e))
                await asyncio.sleep(10)