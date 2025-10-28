"""
homeostasis/__init__.py
======================
Homeostasis subsystem initialization module.

This module initializes the homeostasis components of the Jitter Autopoietic System:
- PID controllers for precise regulation
- Adaptive setpoints for learning and adaptation
- Crisis detection for emergency response
- Configuration validation and integration hooks

The homeostasis subsystem implements the autopoietic system's ability to maintain
physiological stability through feedback control mechanisms.
"""
from .controller import PIDController
from .adaptive_setpoints import AdaptiveSetpoints
from .crisis_detector import CrisisDetector

__all__ = [
    'PIDController',
    'AdaptiveSetpoints',
    'CrisisDetector'
] 