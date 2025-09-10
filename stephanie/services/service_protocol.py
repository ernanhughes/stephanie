# stephanie/services/service_protocol.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

class Service(ABC):
    """Base protocol for all Stephanie services."""
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize service with optional parameters."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return health status and metrics."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Cleanly shut down the service."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Service name for logging and identification."""
        pass