from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from stephanie.constants import SCORABLE_PROCESS, SCORABLE_SUBMIT
from stephanie.scoring.metrics.scorable_processor import ScorableProcessor
from stephanie.scoring.scorable import ScorableFactory

log = logging.getLogger(__name__)


class ScorableProcessorWorker:
    """
    Bus worker that exposes ScorableProcessor over ZmqKnowledgeBus.

    Subjects:
      - scoring.scorable.process  (RPC: request/response)
      - scoring.scorable.submit   (async: fire-and-forget)

    This runs entirely in INLINE mode so it never re-offloads back to the bus.
    """

    SCORABLE_PROCESS = SCORABLE_PROCESS
    SCORABLE_SUBMIT = SCORABLE_SUBMIT

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory,
        container,
        logger,
    ):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.bus = memory.bus
        self.logger = logger

        # Worker-local config: force inline, usually no manifest
        worker_cfg = dict(self.cfg)
        worker_cfg["offload_mode"] = "inline"
        worker_cfg.setdefault("enable_manifest", False)

        self.processor = ScorableProcessor(
            worker_cfg,
            memory=self.memory,
            container=self.container,
            logger=self.logger,
        )

    async def start(self) -> None:
        """
        Attach to the bus and subscribe to scoring subjects.
        Call this once during service startup.
        """
        # Make sure the bus is connected if it exposes that API
        if hasattr(self.bus, "connect") and getattr(self.bus, "is_connected", False) is False:
            await self.bus.connect()

        await self.bus.subscribe(self.SCORABLE_PROCESS, self.handle_rpc)
        await self.bus.subscribe(self.SCORABLE_SUBMIT, self.handle_async)

        log.info(
            "[ScorableProcessorWorker] subscribed rpc=%s async=%s",
            self.SCORABLE_PROCESS,
            self.SCORABLE_SUBMIT,
        )

    # ------------------------------------------------------------------ RPC --

    async def handle_rpc(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        RPC handler for subject=scoring.scorable.process

        Expected envelope from client:
          {
            "job_id": "...",              # added by ZmqKnowledgeBus.request
            "scorable": { ... },
            "context":  { ... },
            "config":   { ... }   # optional, usually ignored in worker
          }

        Returns:
          {
            "status": "ok" | "error",
            "row": { ... },     # when ok
            "error": "...",      # when error
            "job_id": "...",
          }
        """
        scorable_dict = msg.get("scorable") or {}
        context = msg.get("context") or {}
        job_id = msg.get("job_id")

        try:
            scorable = ScorableFactory.from_dict(scorable_dict)
        except Exception as e:
            self.logger.exception("[ScorableProcessorWorker] invalid scorable in rpc: %s", e)
            return {"status": "error", "error": f"invalid scorable: {e}", "job_id": job_id}

        try:
            row = await self.processor.process(scorable, context)
        except Exception as e:
            log.exception("[ScorableProcessorWorker] processing error (rpc): %s", e)
            return {"status": "error", "error": str(e), "job_id": job_id}

        return {"status": "ok", "row": row, "job_id": job_id}

    # ------------------------------------------------------------- ASYNC ----

    async def handle_async(self, msg: Dict[str, Any]) -> None:
        """
        Async handler for subject=scoring.scorable.submit

        Expected envelope from client:
          {
            "scorable": { ... },
            "context":  { ... },
            "config":   { ... }   # optional
          }

        We do the full processing and persistence but return None so that
        ZmqKnowledgeBus does NOT send an RPC reply.
        """
        scorable_dict = msg.get("scorable") or {}
        context = msg.get("context") or {}

        try:
            scorable = ScorableFactory.from_dict(scorable_dict)
        except Exception as e:
            log.exception("[ScorableProcessorWorker] invalid scorable in async: %s", e)
            return None

        try:
            await self.processor.process(scorable, context)
        except Exception as e:
            log.exception("[ScorableProcessorWorker] processing error (async): %s", e)

        # No reply → handler returns None by design
        return None
