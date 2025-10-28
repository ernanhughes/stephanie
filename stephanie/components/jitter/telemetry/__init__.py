"""
telemetry/__init__.py
===================
Telemetry subsystem initialization module.

This module initializes the telemetry components of the Jitter Autopoietic System:
- JAF (Jitter Artifact Format) for legacy preservation
- Vital signs collection and publishing
- Structured logging for debugging and monitoring
- SIS dashboard integration
- Configuration validation and integration hooks

The telemetry subsystem implements the system's ability to monitor, record, and communicate
its internal state to external systems and humans.
"""
from .jaf import JitterArtifactV0
from .telemetry import JASTelemetry
from .structured_logger import StructuredLogger
from .dashboard import JASDashboard

__all__ = [
    'JitterArtifactV0',
    'JASTelemetry',
    'StructuredLogger',
    'JASDashboard'
]