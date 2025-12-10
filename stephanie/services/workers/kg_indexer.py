# stephanie/services/workers/kg_indexer_worker.py
"""
Knowledge Graph Indexer Worker

Listens for 'knowledge_graph.index_request' events from KnowledgeBus.
Processes entity + relationship indexing in the background,
keeping the main pipeline fast and non-blocking.

Supports:
  - Async HNSW indexing
  - Relationship persistence (JSONL)
  - Error recovery & retries
  - Batch processing mode (optional)
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from stephanie.services.knowledge_graph_service import KnowledgeGraphService
from stephanie.utils.retry import retry_with_backoff

log = logging.getLogger(__name__)

class KnowledgeGraphIndexerWorker:
    """
    Asynchronous worker that indexes knowledge graph nodes and edges
    in response to events from the KnowledgeBus.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        logger: Optional[logging.Logger] = None,
        batch_size: int = 1,
        poll_interval: float = 1.0
    ):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger or logging.getLogger(__name__)
        self.batch_size = batch_size
        self.poll_interval = poll_interval

        # Initialize the KG service (in sync mode — we do the actual work here)
        self.kg_service = KnowledgeGraphService(
            cfg=cfg,
            memory=memory,
            logger=self.logger
        )
        self.kg_service.initialize()  # Must be called before use

        # Track stats
        self.stats = {
            "processed": 0,
            "failed": 0,
            "last_event_time": None,
            "uptime_start": datetime.now(timezone.utc).isoformat()
        }

        self.running = False
        self.logger.info("KnowledgeGraphIndexerWorker initialized.", extra={
            "mode": "async-evented",
            "batch_size": self.batch_size,
            "poll_interval": self.poll_interval
        })

    def start(self) -> None:
        """Start the event loop."""
        if self.running:
            return
        self.running = True
        self.logger.info("KnowledgeGraphIndexerWorker started.")

        try:
            while self.running:
                events = self.memory.bus.consume_batch(
                    topic="knowledge_graph.index_request",
                    max_items=self.batch_size
                )
                if not events:
                    time.sleep(self.poll_interval)
                    continue

                for event in events:
                    self._process_event(event)

        except KeyboardInterrupt:
            self.logger.info("KnowledgeGraphIndexerWorker stopped by user.")
        except Exception:
            self.logger.critical("KnowledgeGraphIndexerWorker crashed", exc_info=True)
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        """Graceful shutdown."""
        self.running = False
        self.logger.info("KnowledgeGraphIndexerWorker stopped.", extra=self.stats)

    async def _process_event(self, event: Dict[str, Any]) -> None:
        start_time = time.time()
        payload = event.get("payload", {})
        scorable_id = payload.get("scorable_id", "unknown")

        try:
            if not all(k in payload for k in ["entities", "relationships"]):
                raise ValueError("Missing required fields in index request")

            # Reconstruct domains mapping
            domains = payload.get("domains", [])

            # Add entities
            for ent in payload["entities"]:
                await self._add_entity_with_retry(scorable_id, ent, domains)

            # Add relationships
            for rel in payload["relationships"]:
                await self._add_relationship_with_retry(rel)

            # Log success
            duration = time.time() - start_time
            log.info("KnowledgeGraphIndexSuccess scorable_id %s entities %d relationships %d duration %.2f sec",
                scorable_id,
                len(payload["entities"]),
                len(payload["relationships"]),
                duration
            )
            self.stats["processed"] += 1
            self.stats["last_event_time"] = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            log.warning("KnowledgeGraphIndexFailed scorable_id %s %s %s", scorable_id,
                str(e), traceback.format_exc()
            )
            self.stats["failed"] += 1
            # Optionally publish dead-letter event
            await self._publish_failure_event(event, str(e))

    @retry_with_backoff(max_retries=3, backoff_in_seconds=1)
    async def _add_entity_with_retry(self, scorable_id: str, entity: Dict[str, Any], domains: list) -> None:
        """Add an entity with retry on transient errors."""
        node_id = f"{scorable_id}:{entity['type']}:{entity['start']}-{entity['end']}"
        await self.kg_service._add_entity_node(node_id, entity, domains, scorable_id, "document")

    @retry_with_backoff(max_retries=3, backoff_in_seconds=0.5)
    async def _add_relationship_with_retry(self, rel: Dict[str, Any]) -> None:
        """Add a relationship with retry."""
        await self.kg_service._add_relationship(
            source_id=rel["source"],
            target_id=rel["target"],
            rel_type=rel["type"],
            confidence=rel["confidence"]
        )

    async def _publish_failure_event(self, original_event: Dict[str, Any], error: str) -> None:
        """Send failed job to DLQ or monitoring."""
        failure_event = {
            "event_type": "knowledge_graph.index_failed",
            "payload": {
                "original": original_event,
                "error": error,
                "failed_at": datetime.now(timezone.utc).isoformat()
            }
        }
        try:
            await self.memory.bus.publish(
                subject=failure_event["event_type"],
                payload=failure_event["payload"]
            )

        except Exception as e:
            self.logger.error(f"Failed to publish failure event: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Expose internal metrics."""
        return {
            **self.stats,
            "is_running": self.running,
            "kg_node_count": self.kg_service._stats["total_nodes"],
            "kg_edge_count": self.kg_service._stats["total_edges"],
        } 