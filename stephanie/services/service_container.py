# stephanie/services/service_container.py
"""
Service Container Module

Provides a dependency injection container for managing service lifecycle in the Stephanie AI system.
The container handles service registration, dependency resolution, initialization order, and
graceful shutdown of all services.

Key features:
- Dependency-aware service initialization
- Circular dependency detection
- Health monitoring aggregation
- Ordered shutdown procedure

This container follows the Inversion of Control (IoC) pattern to decouple service creation
from service usage, making the system more modular and testable.
"""

from __future__ import annotations
from typing import Any, Dict, List, Callable
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
    
    def __init__(self):
        """Initialize an empty service container."""
        self._services: Dict[str, Service] = {}  # Initialized service instances
        self._factories: Dict[str, Callable[[], Service]] = {}  # Service factory functions
        self._dependencies: Dict[str, List[str]] = {}  # Service dependency mapping
        self._initialized: set = set()  # Track initialized services
    
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
        """
        Get a service, initializing it and its dependencies if needed.
        
        Args:
            name: Name of the service to retrieve
            
        Returns:
            Initialized service instance
            
        Raises:
            KeyError: If the requested service is not registered
            RuntimeError: If circular dependencies are detected
            ServiceInitializationError: If service fails to initialize
        """
        # Return already initialized service
        if name in self._services:
            return self._services[name]
        
        # Check if service is registered
        if name not in self._factories:
            raise KeyError(f"Service '{name}' is not registered")
        
        # Check for circular dependencies (service is currently initializing)
        if name in self._initialized:
            raise RuntimeError(f"Circular dependency detected for service '{name}'")
        
        # Mark as initializing to detect cycles in dependencies
        self._initialized.add(name)
        
        # Initialize all dependencies first
        for dep_name in self._dependencies.get(name, []):
            self.get(dep_name)
        
        # Create and initialize the service
        service = self._factories[name]()
        service.initialize()
        self._services[name] = service
        
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