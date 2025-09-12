# stephanie/services/knowledge_bus.py
"""
In-process KnowledgeBus
=======================

A durable, dependency-free pub/sub bus for Stephanie.
- Thread-safe
- Supports multiple consumers
- Tracks metrics
- No external dependencies (ideal for local + test environments)
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Optional


class KnowledgeBus:
    """Abstract interface for knowledge event bus."""

    def publish(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError

    def consume(self, topic: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def consume_batch(self, topic: str, max_items: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime metrics for monitoring."""
        raise NotImplementedError


class InProcessKnowledgeBus(KnowledgeBus):
    """
    Thread-safe in-process bus using bounded Python queues.

    Features:
    - Durable per-topic queues (default maxsize = 10,000 events)
    - Multiple producers/consumers safe
    - Metrics tracking
    """

    def __init__(self, maxsize: int = 10_000):
        self._topics: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize

        # Stats
        self._published = 0
        self._consumed = 0
        self._dropped = 0
        self._start_time = time.time()

    def publish(self, event: Dict[str, Any]) -> None:
        if "event_type" not in event or "payload" not in event:
            raise ValueError("Event must include 'event_type' and 'payload'")

        topic = event["event_type"]
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = queue.Queue(maxsize=self._maxsize)
            q = self._topics[topic]

        try:
            q.put_nowait(event)
            self._published += 1
        except queue.Full:
            # Drop oldest (bounded durability)
            try:
                q.get_nowait()
                q.put_nowait(event)
                self._dropped += 1
            except queue.Empty:
                pass  # race condition
            self._published += 1

    def consume(self, topic: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        q = self._topics.get(topic)
        if not q:
            return None
        try:
            event = q.get(timeout=timeout)
            self._consumed += 1
            return event
        except queue.Empty:
            return None

    def consume_batch(self, topic: str, max_items: int = 10) -> List[Dict[str, Any]]:
        q = self._topics.get(topic)
        if not q:
            return []
        items = []
        for _ in range(max_items):
            try:
                event = q.get_nowait()
                items.append(event)
                self._consumed += 1
            except queue.Empty:
                break
        return items

    def get_stats(self) -> Dict[str, Any]:
        """Return live bus metrics."""
        uptime = time.time() - self._start_time
        backlog = {topic: q.qsize() for topic, q in self._topics.items()}
        return {
            "uptime_sec": int(uptime),
            "published": self._published,
            "consumed": self._consumed,
            "dropped": self._dropped,
            "backlog": backlog,
            "topics": list(self._topics.keys())
        }
