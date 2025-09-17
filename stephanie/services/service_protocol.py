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

Key Features:
- Standardized service lifecycle management
- Built-in health monitoring and metrics reporting
- Optional event bus integration for inter-service communication
- Consistent logging and error handling patterns

Usage Example:
    class MyCustomService(Service):
        def __init__(self):
            self._name = "my-custom-service-v1"
            
        def initialize(self, **kwargs):
            # Setup logic here
            pass
            
        # Implement other required methods...
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
        
    Example:
        class DatabaseService(Service):
            @property
            def name(self):
                return "database-postgres-v1"
                
            def initialize(self, connection_string):
                self.connection = connect(connection_string)
                
            def health_check(self):
                return {
                    "status": "healthy" if self.connection.is_connected() else "unhealthy",
                    "timestamp": datetime.now().isoformat()
                }
                
            def shutdown(self):
                self.connection.close()
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
            **kwargs: Service-specific configuration parameters. Common parameters include:
                - config: Service configuration dictionary
                - logger: Dedicated logger instance
                - dependencies: Other services this service depends on
                
        Raises:
            ServiceInitializationError: If service fails to initialize with provided parameters
            
        Example:
            def initialize(self, config=None, logger=None, **kwargs):
                self.config = config or {}
                self.logger = logger or logging.getLogger(self.name)
                # Initialize database connection
                self.db = Database.connect(self.config['database_url'])
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Return comprehensive health status and performance metrics.
        
        This method is called periodically by the service container to monitor
        service health and collect performance metrics.
        
        Returns:
            Dictionary containing health information with standard fields including:
            - status: Overall service status (healthy, degraded, unhealthy)
            - timestamp: Time of health check in ISO format
            - metrics: Service-specific performance metrics
            - dependencies: Status of external dependencies
            
        Example:
            {
                "status": "healthy",
                "timestamp": "2023-10-05T14:30:00Z",
                "metrics": {
                    "response_time_ms": 45.2,
                    "queue_size": 0,
                    "error_rate": 0.01,
                    "uptime_seconds": 3600
                },
                "dependencies": {
                    "database": "connected",
                    "cache": "connected",
                    "api": "degraded"
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
        
        Example:
            def shutdown(self):
                if self.db:
                    self.db.close()
                if self.cache:
                    self.cache.flush()
                self.logger.info("Service shutdown complete")
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

    def set_bus(self, bus: Any) -> None:
        """
        Inject an event bus for inter-service communication.
        
        This method is called by the service container to provide the service
        with access to the system's event bus. Services can use the bus to
        publish events and subscribe to events from other services.
        
        Args:
            bus: The event bus instance implementing publish/subscribe interface
            
        Note:
            This method is optional - services that don't need event bus
            integration can simply ignore it.
        """
        self.bus = bus
        # Try to log the bus attachment, but fail gracefully if logging fails
        lg = getattr(self, "logger", None) or logging.getLogger(self.name)
        try:
            lg.info("ServiceBusAttached", extra={
                "service": self.name, 
                "backend": getattr(bus, "get_backend", lambda: "unknown")()
            })
        except Exception:
            # If logging fails, continue without logging
            pass

    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], Awaitable[None]] | Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to events on the event bus if available.
        
        Args:
            subject: The event subject/topic to subscribe to
            handler: Callback function to handle events. Can be async or sync.
            
        Note:
            This is a no-op if no event bus has been attached to the service.
        """
        bus = getattr(self, "bus", None)
        if not bus:
            return
            
        # Convert sync handlers to async for consistency
        if not asyncio.iscoroutinefunction(handler):
            async def _async_handler(payload: Dict[str, Any]):
                handler(payload)
            return await bus.subscribe(subject, _async_handler)
            
        return await bus.subscribe(subject, handler)

    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """
        Publish an event to the event bus if available.
        
        Args:
            subject: The event subject/topic to publish to
            payload: Dictionary containing event data
            
        Note:
            This is a no-op if no event bus has been attached to the service.
        """
        bus = getattr(self, "bus", None)
        if not bus:
            return
            
        await bus.publish(subject, payload)