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

import logging
import inspect
from typing import Any, Callable, Dict, List

from stephanie.services.service_protocol import Service
from stephanie.utils.secret_utils import redact_secrets

logger = logging.getLogger(__name__)

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

    def __init__(self, cfg: Dict[str, Any], memory: Any, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger or getattr(memory, "logger", None) or logging.getLogger(__name__)
        self._factories: Dict[str, Callable[[], Service]] = {}
        self._services: Dict[str, Service] = {}
        self._dependencies: Dict[
            str, List[str]
        ] = {}  # Service dependency mapping
        self._initialized: set = set()  # Track initialized services
        self._order: List[str] = []
        self._initializing: List[str] = []
        # MemoryTool owns the configured bus. The service container must not
        # construct a competing bus with a different lifecycle/configuration.
        self._bus = getattr(memory, "bus", None)
        self._init_args: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        if self._bus:
            connected = await self._bus.connect()
            if not connected:
                bus_cfg = self.cfg.get("bus", {})
                if bus_cfg.get("required", bus_cfg.get("strict", False)):
                    raise RuntimeError(
                        "Required event bus failed to connect; verify bus configuration and availability"
                    )
                self.logger.warning("Optional event bus failed to connect")
            else:
                self.logger.info(
                    f"Connected to event bus backend: {self._bus.get_backend()}"
                )

        # Then initialize services
        for name in self._factories.keys():
            await self.get_async(name)

    def register(
        self,
        name: str,
        factory: Callable[[], Service],
        dependencies: List[str] = None,
        init_args: Dict[str, Any] = None,
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
        self._init_args[name] = init_args or {}



    def _assert_registered(self, name: str) -> None:
        if name in self._initializing:
            cycle = " -> ".join([*self._initializing, name])
            raise RuntimeError(f"Circular service dependency: {cycle}")
        if name not in self._factories:
            raise KeyError(f"Service '{name}' is not registered")

    def _new_service(self, name: str) -> Service:
        service = self._factories[name]()
        if hasattr(service, "set_bus"):
            service.set_bus(self._bus)
        elif hasattr(service, "bus"):
            service.bus = self._bus
        return service

    def _init_kwargs(self, name: str) -> Dict[str, Any]:
        init_kwargs = self._init_args.get(name, {}) or {}
        if callable(init_kwargs):
            init_kwargs = init_kwargs()
        return init_kwargs

    def get(self, name: str) -> Service:
        """Synchronously get a service; async lifecycle services use get_async()."""
        if name in self._services:
            return self._services[name]
        self._assert_registered(name)
        self._initializing.append(name)
        try:
            for dep in self._dependencies.get(name, []):
                if dep.startswith("?"):
                    if dep[1:] in self._factories:
                        self.get(dep[1:])
                else:
                    self.get(dep)

            service = self._new_service(name)
            result = service.initialize(**self._init_kwargs(name))
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise RuntimeError(
                    f"Service '{name}' has async initialization; use await container.get_async('{name}')"
                )
            self._services[name] = service
            self._initialized.add(name)
            self._order.append(name)
            return service
        finally:
            self._initializing.pop()

    def get_service(self, name: str) -> Service:
        """Backward-compatible accessor used by existing service consumers."""
        return self.get(name)

    def get_or_none(self, name: str) -> Service | None:
        """Return an initialized/registered service, or ``None`` if absent."""
        if name not in self._factories:
            return None
        return self.get(name)

    async def get_async(self, name: str) -> Service:
        """Get a service and await its initialization when it is asynchronous."""
        if name in self._services:
            return self._services[name]
        self._assert_registered(name)
        self._initializing.append(name)
        try:
            for dep in self._dependencies.get(name, []):
                if dep.startswith("?"):
                    if dep[1:] in self._factories:
                        await self.get_async(dep[1:])
                else:
                    await self.get_async(dep)

            service = self._new_service(name)
            result = service.initialize(**self._init_kwargs(name))
            if inspect.isawaitable(result):
                await result
            self._services[name] = service
            self._initialized.add(name)
            self._order.append(name)
            return service
        finally:
            self._initializing.pop()

    def shutdown_all(self):
        """
        Cleanly shut down all services in reverse initialization order.

        Services are shut down in reverse order of initialization to ensure
        dependencies are available until their dependents are shut down.

        Any exceptions during shutdown are caught and logged but don't interrupt
        the shutdown process for other services.
        """
        # Shutdown in reverse initialization order (dependents before dependencies)
        for name in reversed(self._order):
            try:
                result = self._services[name].shutdown()
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    logger.error(
                        "Async service '%s' requires await container.shutdown()",
                        name,
                    )
            except Exception as e:
                logger.error(
                    "Error shutting down service '%s': %s",
                    name,
                    redact_secrets(str(e)),
                )

        # Clear all state
        self._services.clear()
        self._initialized.clear()
        self._order.clear()

    async def shutdown(self):
        """Gracefully shut down all services and the bus."""
        # Shutdown in reverse initialization order
        for name in reversed(self._order):
            try:
                if hasattr(self._services[name], "shutdown"):
                    result = self._services[name].shutdown()
                    if inspect.isawaitable(result):
                        await result
            except Exception as e:
                logger.error(
                    "Error shutting down %s: %s", name, redact_secrets(str(e))
                )

        # Shutdown the bus last
        if self._bus:
            await self._bus.close()

        self._services.clear()
        self._initialized.clear()
        self._order.clear()

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
        report = {}
        for name, service in self._services.items():
            result = service.health_check()
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                report[name] = {
                    "status": "async_health_check_requires_await"
                }
            else:
                report[name] = result
        return report

    async def health_report_async(self) -> Dict[str, Dict[str, Any]]:
        """Collect health checks from synchronous and asynchronous services."""
        report = {}
        for name, service in self._services.items():
            result = service.health_check()
            if inspect.isawaitable(result):
                result = await result
            report[name] = result
        return report
