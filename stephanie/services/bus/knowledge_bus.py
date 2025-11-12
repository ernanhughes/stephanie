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
