# stephanie/services/workers/nexus_feature_worker.py
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from stephanie.scoring.metrics.scorable_processor import ScorableProcessor
from stephanie.services.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


class NexusFeatureWorker(BaseWorker):
    """
    Worker that runs ScorableProcessor on incoming scorables and mirrors
    metrics/embeddings into the Nexus store.

    Default subject: 'nexus.features.index_request'

    Envelope schema (via bus or inline):

        {
          "event_type": "nexus.features.index_request",  # optional
          "payload": {
            "scorable":  {...},           # single scorable dict
            # or
            "scorables": [{...}, {...}],  # batch
            "context":   {...}            # pipeline / run context
          }
        }

    Every node in the graph is a Scorable → this worker attaches:

      * NexusScorable row (text, domains, entities, meta)
      * NexusEmbedding row (embed_global)
      * NexusMetrics row (metrics_columns / values / vector)

    Later we can materialize “feature edges” from these metrics.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger,
    ) -> None:
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)

        worker_cfg = cfg.get("worker", {})
        self.subject = worker_cfg.get(
            "subject",
            "nexus.features.index_request",
        )
        self.debug_rows = bool(worker_cfg.get("debug_rows", False))

        self.stats: Dict[str, Any] = {
            "processed": 0,
            "failed": 0,
            "last_event_time": None,
        }

        self.nexus_store = None
        self.processor: Optional[ScorableProcessor] = None

    # ------------------------ BaseWorker hooks ----------------------- #

    async def init_services(self) -> None:
        """Resolve NexusStore + ScorableProcessor from memory/container."""
        # Nexus store: memory.nexus is the canonical adapter
        self.nexus_store = self.memory.nexus
        proc_cfg = self.cfg.get("scorable_processor", {})
        self.processor = ScorableProcessor(
            proc_cfg,
            memory=self.memory,
            container=self.container,
            logger=self.logger,
        )
        log.info(
            "[NexusFeatureWorker] initialized",
            extra={"subject": self.subject},
        )

    async def register_subjects(self) -> None:
        """Subscribe to feature-index requests on the bus."""
        await self.subscribe(self.subject, self.handle_job)

    async def worker_health(self) -> Dict[str, Any]:
        """Expose basic metrics for dashboards."""
        return {
            "subject": self.subject,
            "processed": self.stats["processed"],
            "failed": self.stats["failed"],
            "last_event_time": self.stats["last_event_time"],
        }

    # ----------------------------- core ------------------------------ #

    async def handle_job(self, envelope: Dict[str, Any]) -> None:
        """
        Main handler.

        You can call this via the bus OR inline, as long as you pass
        the same envelope shape.
        """
        t0 = time.time()
        payload = envelope.get("payload") or envelope

        context: Dict[str, Any] = payload.get("context") or {}
        run_id: Optional[str] = context.get("pipeline_run_id")

        # Accept both single + batch payloads
        scorable_dicts: List[Dict[str, Any]] = []
        if "scorables" in payload:
            scorable_dicts.extend(payload.get("scorables") or [])
        if "scorable" in payload and payload.get("scorable"):
            scorable_dicts.append(payload["scorable"])

        if not scorable_dicts:
            log.debug("[NexusFeatureWorker] empty payload; nothing to do")
            return

        if not self.processor:
            log.error("[NexusFeatureWorker] processor not initialized")
            return

        try:
            # 1) Run all features via ScorableProcessor
            rows = await self.processor.process_many(scorable_dicts, context)
            n = len(rows)

            # 2) Mirror into Nexus (if available)
            if self.nexus_store is None:
                log.warning(
                    "[NexusFeatureWorker] processed %d scorables but "
                    "no nexus_store is configured",
                    n,
                )
                return

            for row in rows:
                self._write_row_to_nexus(row, run_id)

            self.stats["processed"] += n
            self.stats["last_event_time"] = datetime.now(timezone.utc).isoformat()

            if self.debug_rows and rows:
                first = rows[0]
                log.info(
                    "[NexusFeatureWorker] processed=%d first_id=%s "
                    "metrics=%s",
                    n,
                    first.get("scorable_id"),
                    list((first.get("metrics_vector") or {}).keys())[:8],
                )

            log.debug(
                "[NexusFeatureWorker] done batch n=%d in %.2f ms",
                n,
                (time.time() - t0) * 1000.0,
            )

        except Exception:
            self.stats["failed"] += len(scorable_dicts)
            log.warning(
                "[NexusFeatureWorker] handle_job failed for %d scorables: %s",
                len(scorable_dicts),
                traceback.format_exc(),
            )

    # -------------------------- internals ---------------------------- #

    def _write_row_to_nexus(self, row: Dict[str, Any], run_id: Optional[str]) -> None:
        """
        Take the ScorableProcessor row dict and fan it out into:

          * nexus_scorables
          * nexus_embeddings
          * nexus_metrics

        so that *every graph node* (section / snippet / etc.) has a
        feature vector we can later turn into feature-edges + VPM tiles.
        """
        if self.nexus_store is None:
            return

        scorable_id = row.get("scorable_id")
        if not scorable_id:
            log.debug(
                "[NexusFeatureWorker] row has no scorable_id, skipping: %s",
                row,
            )
            return

        # --------- 1) core scorable row --------------------------------
        scorable_row = {
            "id": scorable_id,
            "chat_id": row.get("conversation_id"),
            "turn_index": row.get("order_index"),
            "target_type": row.get("scorable_type"),
            "text": row.get("text") or "",
            "domains": row.get("domains") or [],
            "entities": row.get("ner") or [],
            "meta": {
                "title": row.get("title"),
                "external_id": row.get("external_id"),
                "ai_score": row.get("ai_score"),
                "star": row.get("star"),
                "goal_ref": row.get("goal_ref"),
                "metrics": row.get("metrics_vector") or {},
                "run_id": run_id,
            },
        }

        try:
            self.nexus_store.upsert_scorable(scorable_row)
        except Exception as e:
            log.exception(
                "[NexusFeatureWorker] upsert_scorable failed id=%s: %s",
                scorable_id,
                e,
            )

        # --------- 2) metrics vector ----------------------------------
        metrics_cols = row.get("metrics_columns") or []
        metrics_vals = row.get("metrics_values") or []
        metrics_vec = row.get("metrics_vector") or {}

        if metrics_cols and metrics_vals:
            try:
                self.nexus_store.upsert_metrics(
                    scorable_id=scorable_id,
                    columns=metrics_cols,
                    values=metrics_vals,
                    vector=metrics_vec,
                )
            except Exception as e:
                log.exception(
                    "[NexusFeatureWorker] upsert_metrics failed id=%s: %s",
                    scorable_id,
                    e,
                )

        # --------- 3) global embedding (for retrieval / geometry) -----
        embed = row.get("embed_global")
        if embed:
            try:
                # NOTE: guard against the earlier .astype(list) crash
                vec = np.array(embed, dtype=float)
                self.nexus_store.upsert_embedding(
                    scorable_id=scorable_id,
                    vec=vec,
                    store_norm=True,
                )
            except Exception as e:
                log.exception(
                    "[NexusFeatureWorker] upsert_embedding failed id=%s: %s",
                    scorable_id,
                    e,
                )

        # --------- 4) future: feature-edges ----------------------------
        # At this point we have:
        #   - domains (list[str])
        #   - entities (NER)
        #   - metrics_vector (dict[str, float])
        #
        # You can later create edges like:
        #   (scorable_id) -[:HAS_DOMAIN]-> ("domain:AI")
        #   (scorable_id) -[:HAS_METRIC]-> ("metric:novelty>0.8")
        #
        # using self.nexus_store.write_edges(run_id, edges=...)
        # We'll keep that as a follow-up so we can design good schemas.
