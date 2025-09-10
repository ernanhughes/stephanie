# stephanie/services/service_container.py
from __future__ import annotations

from typing import Any, Dict, List, Callable
from stephanie.services.service_protocol import Service

class ServiceContainer:
    """Manages service lifecycle and dependencies with proper initialization order."""
    
    def __init__(self):
        self._services: Dict[str, Service] = {}
        self._factories: Dict[str, Callable[[], Service]] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._initialized: set = set()
    
    def register(
        self, 
        name: str, 
        factory: Callable[[], Service], 
        dependencies: List[str] = None
    ):
        """Register a service factory with its dependencies."""
        self._factories[name] = factory
        self._dependencies[name] = dependencies or []
    
    def get(self, name: str) -> Service:
        """Get a service, initializing it and its dependencies if needed."""
        if name in self._initialized:
            return self._services[name]
        
        # Check for circular dependencies
        if name in self._initialized:
            raise RuntimeError(f"Circular dependency detected for service '{name}'")
        
        # Initialize dependencies first
        self._initialized.add(name)  # Mark as initializing to detect cycles
        for dep in self._dependencies.get(name, []):
            self.get(dep)
        self._initialized.remove(name)
        
        # Create and store the service
        self._services[name] = self._factories[name]()
        self._initialized.add(name)
        return self._services[name]
    
    def shutdown_all(self):
        """Cleanly shut down all services in reverse initialization order."""
        # Shutdown in reverse initialization order
        for name in reversed(list(self._initialized)):
            try:
                self._services[name].shutdown()
            except Exception as e:
                print(f"Error shutting down {name}: {e}")
        self._services.clear()
        self._initialized.clear()
    
    def health_report(self) -> Dict[str, Dict[str, Any]]:
        """Generate health report for all services."""
        return {
            name: service.health_check() 
            for name, service in self._services.items()
        }