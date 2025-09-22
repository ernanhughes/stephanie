# stephanie/services/service_container.py
"""
Service Container Module

Provides a dependency injection container for managing service lifecycle in the Stephanie AI system.
The container handles service registration, dependency resolution, initialization order, and
graceful shutdown of all services.
Oh see great in Chinese as well
Key features:
- Dependency-aware service initialization
- Circular dependency detection
- Health monitoring aggregation
- Ordered shutdown procedure

This container follows the Inversion of Control (IoC) pattern to decouple service creation
from service usage, making the system more modular and testable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from stephanie.services.bus.hybrid_bus import HybridKnowledgeBus
from stephanie.services.service_protocol import Service


class ServiceContainer:
    """
    Manages service lifecycle and dependencies with proper initialization order.
    
    The ServiceContainer is responsible for:
    1. Registering service factories with their dependencies
    2. Resolving and initializing services in the correct order
    3. Detecting and preventing circular dependencies
    4. Providing access to initialized services
    5. Coordinating graceful shutdown of all services
    6. Aggregating health checks across all services
    
    Services are initialized on-demand when first requested, after all their
    dependencies have been initialized.
    
    Example:
        container = ServiceContainer()
        container.register('database', DatabaseService, ['config'])
        container.register('config', ConfigService)
        db_service = container.get('database')  # Initializes config first, then database
    """
    
    def __init__(self, cfg: Dict[str, Any], logger=None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger("services")
        self._factories: Dict[str, Callable[[], Service]] = {}
        self._services: Dict[str, Service] = {}
        self._dependencies: Dict[str, List[str]] = {}  # Service dependency mapping
        self._initialized: set = set()  # Track initialized services        
        self._order: List[str] = []
        self._bus = None

    async def initialize(self):
        self._bus = HybridKnowledgeBus(self.cfg.get("bus", {}), self.logger)
        connected =await self._bus.connect()
        if not connected:
            self.logger.error("Failed to connect to event bus")
        else:
            self.logger.info(f"Connected to event bus backend: {self._bus.get_backend()}")
        
        # Then initialize services
        for name in self._factories.keys():
            self.get(name)

    def register(
        self, 
        name: str, 
        factory: Callable[[], Service], 
        dependencies: List[str] = None
    ):
        """
        Register a service factory with its dependencies.
        
        Args:
            name: Unique identifier for the service
            factory: Callable that returns a Service instance when called
            dependencies: List of service names that must be initialized before this service
            
        Raises:
            ValueError: If a service with the same name is already registered
        """
        if name in self._factories:
            raise ValueError(f"Service '{name}' is already registered")
            
        self._factories[name] = factory
        self._dependencies[name] = dependencies or []
    
    def get(self, name: str) -> Service:
        """Get a service, initializing it and its dependencies."""
        if name in self._services:
            return self._services[name]
            
        if name not in self._factories:
            raise KeyError(f"Service '{name}' is not registered")
            
        # Initialize dependencies first
        for dep in self._dependencies.get(name, []):
            self.get(dep)
            
        # Create service
        service = self._factories[name]()
        
        # Inject the bus if the service supports it
        if hasattr(service, 'set_bus'):
            service.set_bus(self._bus)
        elif hasattr(service, 'bus'):
            service.bus = self._bus
            
        # Initialize the service
        service.initialize()
        self._services[name] = service
        self._initialized.add(name)
        
        return service
    
    def shutdown_all(self):
        """
        Cleanly shut down all services in reverse initialization order.
        
        Services are shut down in reverse order of initialization to ensure
        dependencies are available until their dependents are shut down.
        
        Any exceptions during shutdown are caught and logged but don't interrupt
        the shutdown process for other services.
        """
        # Shutdown in reverse initialization order (dependents before dependencies)
        for name in reversed(list(self._initialized)):
            try:
                self._services[name].shutdown()
            except Exception as e:
                # Use proper logging in production code
                print(f"Error shutting down service '{name}': {e}")
        
        # Clear all state
        self._services.clear()
        self._initialized.clear()

    async def shutdown(self):
        """Gracefully shut down all services and the bus."""
        # Shutdown in reverse initialization order
        for name in reversed(list(self._initialized)):
            try:
                if hasattr(self._services[name], 'shutdown'):
                    await self._services[name].shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down {name}: {str(e)}")
                
        # Shutdown the bus last
        if self._bus:
            await self._bus.close()
            
        self._services.clear()
        self._initialized.clear()
    
    def health_report(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate a comprehensive health report for all initialized services.
        
        Returns:
            Dictionary mapping service names to their health status reports
            
        Example:
            {
                "database": {
                    "status": "healthy",
                    "metrics": {"connection_count": 5}
                },
                "cache": {
                    "status": "degraded",
                    "metrics": {"hit_rate": 0.85}
                }
            }
        """
        return {
            name: service.health_check() 
            for name, service in self._services.items()
        }