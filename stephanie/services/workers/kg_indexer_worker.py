# stephanie/services/workers/kg_indexer_worker.py

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from stephanie.services.knowledge_graph_service import KnowledgeGraphService
from stephanie.services.workers.base_worker import BaseWorker
from stephanie.utils.retry import retry_with_backoff

log = logging.getLogger(__name__)


class KnowledgeGraphIndexerWorker(BaseWorker):
    """
    Asynchronous worker that indexes knowledge graph nodes and edges
    in response to events from the KnowledgeBus.

    Listens on subject: 'knowledge_graph.index_request'

    Envelope schema (per event):

        {
          "event_type": "knowledge_graph.index_request",  # optional
          "payload": {
            "scorable_id": "...",
            "domains": [...],          # optional
            "entities": [...],         # required
            "relationships": [...],    # required
          }
        }
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Any,
    ) -> None:
        # BaseWorker wires cfg / memory / container / logger
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)

        # Subject can be overridden in cfg["worker"]["subject"]
        wcfg = self.cfg.get("worker", {})
        self.subject: str = wcfg.get(
            "subject",
            "knowledge_graph.index_request",
        )

        # Created in init_services()
        self.kg_service: Optional[KnowledgeGraphService] = None

        # Extend base stats with KG-specific counters
        self._stats.update(
            {
                "kg_processed": 0,
                "kg_failed": 0,
                "kg_last_event_time": None,
            }
        )

    # ------------------------ BaseWorker hooks ----------------------- #

    async def init_services(self) -> None:
        """Instantiate the KnowledgeGraphService."""
        self.kg_service = KnowledgeGraphService(
            cfg=self.cfg,
            memory=self.memory,
            logger=self.logger,
        )
        self.kg_service.initialize()

        log.info(
            "KnowledgeGraphIndexerWorker initialized.",
            extra={
                "mode": "bus-subscribe",
                "subject": self.subject,
            },
        )

    async def register_subjects(self) -> None:
        """Subscribe to bus subjects."""
        await self.subscribe(self.subject, self.handle_job)

    async def worker_health(self) -> Dict[str, Any]:
        """Expose basic KG health along with base stats."""
        total_nodes = 0
        total_edges = 0
        if self.kg_service is not None:
            total_nodes = self.kg_service._stats.get("total_nodes", 0)
            total_edges = self.kg_service._stats.get("total_edges", 0)

        return {
            "kg_node_count": total_nodes,
            "kg_edge_count": total_edges,
        }

    # --------------------------- core logic -------------------------- #

    async def handle_job(self, envelope: Dict[str, Any]) -> None:
        """
        Bus callback for 'knowledge_graph.index_request'.
        """
        start_time = time.time()
        payload = envelope.get("payload") or {}
        scorable_id = payload.get("scorable_id", "unknown")

        try:
            if not all(k in payload for k in ("entities", "relationships")):
                raise ValueError("Missing required fields in index request")

            domains = payload.get("domains") or []

            # Index entities
            for ent in payload["entities"]:
                await self._add_entity_with_retry(scorable_id, ent, domains)

            # Index relationships
            for rel in payload["relationships"]:
                await self._add_relationship_with_retry(rel)

            duration = time.time() - start_time
            log.info(
                "KnowledgeGraphIndexSuccess scorable_id=%s entities=%d relationships=%d duration=%.2fs",
                scorable_id,
                len(payload["entities"]),
                len(payload["relationships"]),
                duration,
            )

            self._stats["kg_processed"] += 1
            self._stats["kg_last_event_time"] = datetime.now(
                timezone.utc
            ).isoformat()

        except Exception as e:
            log.warning(
                "KnowledgeGraphIndexFailed scorable_id=%s error=%s\n%s",
                scorable_id,
                str(e),
                traceback.format_exc(),
            )
            self._stats["kg_failed"] += 1
            await self._publish_failure_event(envelope, str(e))

    # ----------------------- helpers with retry ---------------------- #

    @retry_with_backoff(max_retries=3, backoff_in_seconds=1.0)
    async def _add_entity_with_retry(
        self,
        scorable_id: str,
        entity: Dict[str, Any],
        domains: list,
    ) -> None:
        """
        Add an entity node, tagged with its parent scorable and domains.

        Uses KnowledgeGraphService, which (after the Nexus wiring changes)
        will also mirror nodes into Nexus via NexusStore.
        """
        if self.kg_service is None:
            raise RuntimeError("kg_service is not initialized")

        node_id = f"{scorable_id}:{entity['type']}:{entity['start']}-{entity['end']}"
        await self.kg_service._add_entity_node(
            node_id,
            entity,
            domains,
            scorable_id,
            "document",
        )

    @retry_with_backoff(max_retries=3, backoff_in_seconds=0.5)
    async def _add_relationship_with_retry(self, rel: Dict[str, Any]) -> None:
        """Add a relationship with retry."""
        if self.kg_service is None:
            raise RuntimeError("kg_service is not initialized")

        await self.kg_service._add_relationship(
            source_id=rel["source"],
            target_id=rel["target"],
            rel_type=rel["type"],
            confidence=rel.get("confidence", 0.9),
        )

    async def _publish_failure_event(
        self,
        original_envelope: Dict[str, Any],
        error: str,
    ) -> None:
        """Send failed job to DLQ or monitoring."""
        failure_event = {
            "event_type": "knowledge_graph.index_failed",
            "payload": {
                "original": original_envelope,
                "error": error,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        try:
            await self.publish(
                subject=failure_event["event_type"],
                payload=failure_event["payload"],
            )
        except Exception as e:
            # Don't let DLQ issues kill the worker
            log.error("Failed to publish KG index failure event: %s", str(e))

    # ----------------------------- misc ------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        """
        Convenience accessor for the worker stats from callers that don't
        integrate with the health loop.
        """
        out = dict(self._stats)
        if self.kg_service is not None:
            out["kg_node_count"] = self.kg_service._stats.get("total_nodes", 0)
            out["kg_edge_count"] = self.kg_service._stats.get("total_edges", 0)
        return out
