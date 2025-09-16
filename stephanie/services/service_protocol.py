# stephanie/services/service_protocol.py
"""
Service Protocol Module

Defines the abstract base class (protocol) for all services in the Stephanie AI system.
This ensures consistent interface implementation across all service types including:
- Knowledge services
- Scoring services
- Reasoning services
- Memory services
- Integration services

The Service protocol establishes the mandatory interface that all services must implement
for proper integration with Stephanie's core system and service orchestration framework.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Optional


class Service(ABC):
    """
    Abstract base class defining the protocol for all Stephanie services.
    
    This protocol ensures all services implement a consistent interface for:
    - Initialization and configuration
    - Health monitoring and metrics reporting
    - Graceful shutdown procedures
    - Service identification
    
    All concrete service implementations must inherit from this class and
    implement the abstract methods defined in this protocol.
    
    Attributes:
        name (str): Read-only property returning the service name for logging and identification
    """
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize service with optional configuration parameters.
        
        This method is called once when the service is first created and should handle:
        - Resource allocation
        - Connection establishment
        - Configuration validation
        - Precomputation setup
        
        Args:
            **kwargs: Service-specific configuration parameters
            
        Raises:
            ServiceInitializationError: If service fails to initialize with provided parameters
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Return comprehensive health status and performance metrics.
        
        Returns:
            Dictionary containing health information with standard fields including:
            - status: Overall service status (healthy, degraded, unhealthy)
            - timestamp: Time of health check
            - metrics: Service-specific performance metrics
            - dependencies: Status of external dependencies
            
        Example:
            {
                "status": "healthy",
                "timestamp": "2023-10-05T14:30:00Z",
                "metrics": {
                    "response_time_ms": 45.2,
                    "queue_size": 0,
                    "error_rate": 0.01
                },
                "dependencies": {
                    "database": "connected",
                    "cache": "connected"
                }
            }
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Cleanly shut down the service and release all resources.
        
        This method should ensure:
        - All connections are properly closed
        - Temporary resources are cleaned up
        - In-progress operations are completed or safely terminated
        - Any persistent state is properly saved
        
        The service should be designed to be restarted after shutdown.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Service name for logging, identification and service discovery.
        
        Returns:
            Unique string identifier for the service type
            
        Note:
            Service names should follow the pattern: "domain-purpose-version"
            Example: "knowledge-ner-v1", "scoring-relevance-v2"
        """
        pass


        # --- New: bus awareness (optional, progressive) ---
    def set_bus(self, bus: Any) -> None:
        """Inject an event bus. Optional for services that use events."""
        self.bus = bus
        # optional logger if service provides one
        lg = getattr(self, "logger", None) or logging.getLogger(self.name)
        try:
            lg.info("ServiceBusAttached", extra={"service": self.name, "backend": getattr(bus, "get_backend", lambda: "unknown")()})
        except Exception:
            pass

    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], Awaitable[None]] | Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe if a bus exists; no-op otherwise."""
        bus = getattr(self, "bus", None)
        if not bus:
            return
        # wrap sync handlers
        if not asyncio.iscoroutinefunction(handler):
            async def _async_handler(payload: Dict[str, Any]):
                handler(payload)
            return await bus.subscribe(subject, _async_handler)
        return await bus.subscribe(subject, handler)

    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """Publish if a bus exists; no-op otherwise."""
        bus = getattr(self, "bus", None)
        if not bus:
            return
        await bus.publish(subject, payload)
