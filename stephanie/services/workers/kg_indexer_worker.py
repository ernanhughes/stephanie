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
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)

        self.batch_size = cfg.get("batch_size", 1)
        self.poll_interval = cfg.get("poll_interval", 1.0)
        self.debug_tree = bool(self.cfg.get("debug_tree", True))
        self.kg_service: KnowledgeGraphService = KnowledgeGraphService(
            cfg=cfg,
            memory=memory,
            logger=log,
        )
        self.kg_service.initialize()

        self.stats = {
            "processed": 0,
            "failed": 0,
            "last_event_time": None,
            "uptime_start": datetime.now(timezone.utc).isoformat(),
        }

        self.running = False
        self.subject = "knowledge_graph.index_request"

        # NEW: enable/disable noisy tree logs via config
        worker_cfg = cfg.get("worker", {})
        self.debug_tree = bool(worker_cfg.get("debug_tree", False))

        log.info(
            "KnowledgeGraphIndexerWorker initialized.",
            extra={
                "mode": "bus-subscribe",
                "subject": self.subject,
                "debug_tree": self.debug_tree,
            },
        )

    # ------------------------ BaseWorker hooks ----------------------- #

    async def init_services(self) -> None:
        """Instantiate the KnowledgeGraphService."""
        self.kg_service = KnowledgeGraphService(
            cfg=self.cfg,
            memory=self.memory,
            logger=log,
        )
        self.kg_service.initialize()

        log.info(
            "KnowledgeGraphIndexerWorker initialized.mode bus-subscribe subject=%s",
            self.subject,
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
        Bus callback for subject='knowledge_graph.index_request'.
        """
        start_time = time.time()
        payload = envelope.get("payload") or {}
        scorable_id = payload.get("scorable_id", "unknown")

        # Snapshot counts *before* we index, so we can log deltas
        before_nodes = self.kg_service._stats.get("total_nodes", 0)
        before_edges = self.kg_service._stats.get("total_edges", 0)

        try:
            if not all(k in payload for k in ["entities", "relationships"]):
                raise ValueError("Missing required fields in index request")

            domains = payload.get("domains") or []

            # Index entities
            for ent in payload["entities"]:
                await self._add_entity_with_retry(scorable_id, ent, domains)

            # Index relationships
            for rel in payload["relationships"]:
                await self._add_relationship_with_retry(rel)

            duration = time.time() - start_time

            after_nodes = self.kg_service._stats.get("total_nodes", 0)
            after_edges = self.kg_service._stats.get("total_edges", 0)
            delta_nodes = after_nodes - before_nodes
            delta_edges = after_edges - before_edges

            # Compact growth summary
            log.info(
                "KGIndexSuccess scorable_id=%s entities=%d relationships=%d "
                "duration=%.2fs nodes(+%d->%d) edges(+%d->%d)",
                scorable_id,
                len(payload["entities"]),
                len(payload["relationships"]),
                duration,
                delta_nodes,
                after_nodes,
                delta_edges,
                after_edges,
            )

            # Optional: pretty tree view for this event
            if self.debug_tree:
                tree = self._format_event_tree(scorable_id, payload)
                # One multi-line log entry so it reads as a block in the log
                log.info("\n%s", tree)

            log.info(
                "KnowledgeGraphIndexSuccess scorable_id=%s entities=%d relationships=%d duration=%.2f sec",
                scorable_id,
                len(payload["entities"]),
                len(payload["relationships"]),
                duration,
            )
            self.stats["processed"] += 1
            self.stats["last_event_time"] = datetime.now(timezone.utc).isoformat()

            if self.debug_tree:
                try:
                    self.kg_service.log_local_tree(
                        scorable_id=scorable_id,
                        max_entities=self.cfg.get("debug_tree_max_entities", 8),
                        max_edges_per_entity=self.cfg.get("debug_tree_max_edges_per_entity", 6),
                    )
                except Exception:
                    log.exception(
                        "Failed to log local KG tree",
                        extra={"scorable_id": scorable_id},
                    )
        except Exception as e:
            log.warning(
                "KnowledgeGraphIndexFailed scorable_id=%s error=%s\n%s",
                scorable_id,
                str(e),
                traceback.format_exc(),
            )
            self.stats["failed"] += 1
            await self._publish_failure_event(envelope, str(e))

    # ----------------------- helpers with retry ---------------------- #

    @retry_with_backoff(max_retries=3, backoff_in_seconds=1)
    async def _add_entity_with_retry(
        self,
        scorable_id: str,
        entity: Dict[str, Any],
        domains: list,
    ) -> None:
        """
        Add an entity node, tagged with its parent scorable and domains.
        """
        node_id = f"{scorable_id}:{entity['type']}:{entity['start']}-{entity['end']}"
        await self.kg_service._add_entity_node(
            node_id,
            entity,
            domains,
            scorable_id,
            "document",
            meta={},  # or pass through extra meta from payload if you like
        )

    @retry_with_backoff(max_retries=3, backoff_in_seconds=0.5)
    async def _add_relationship_with_retry(self, rel: Dict[str, Any]) -> None:
        """Add a relationship with retry."""
        await self.kg_service._add_relationship(
            source_id=rel["source"],
            target_id=rel["target"],
            rel_type=rel["type"],
            confidence=rel.get("confidence", 0.9),
        )

    # ------------------------ pretty tree view ----------------------- #

    def _format_event_tree(self, scorable_id: str, payload: Dict[str, Any]) -> str:
        """
        Build a small ASCII "tree" for this index event, purely from the payload.

        This doesn't query Nexus or HNSW – it's just a visual summary of what
        we *just* added, so you can watch the graph grow in the log.
        """
        lines: list[str] = []

        lines.append(f"KGIndex[{scorable_id}]")

        entities = payload.get("entities") or []
        relationships = payload.get("relationships") or []

        # Entities under the root
        for ent in entities:
            etype = str(ent.get("type", "UNKNOWN"))
            text = str(ent.get("text", ""))[:80].replace("\n", " ")
            lines.append(f"  • entity {etype:8s}: {text}")

        # Relationships block
        if relationships:
            lines.append("  relationships:")
            for rel in relationships:
                r_type = rel.get("type", "REL")
                src = rel.get("source", "?")
                tgt = rel.get("target", "?")
                conf = rel.get("confidence", 0.9)
                lines.append(
                    f"    - {src} -[{r_type}]-> {tgt} (conf={conf:.2f})"
                )

        return "\n".join(lines)

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

    async def start(self) -> None:
        """Attach to the bus and subscribe to index events."""
        if self.running:
            return
        self.running = True

        if hasattr(self.memory.bus, "connect") and getattr(self.memory.bus, "is_connected", False) is False:
            await self.memory.bus.connect()

        await self.memory.bus.subscribe(self.subject, self.handle_job)
        log.info(
            "KnowledgeGraphIndexerWorker started and subscribed.",
            extra={"subject": self.subject},
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        if not self.running:
            return
        self.running = False
        try:
            await self.memory.bus.unsubscribe(self.subject, self.handle_job)
        except Exception:
            pass

        log.info("KnowledgeGraphIndexerWorker stopped.", extra=self.stats)