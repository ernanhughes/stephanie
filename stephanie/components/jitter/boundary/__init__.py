"""
boundary/__init__.py
====================
Boundary subsystem initialization module.

This module initializes the boundary components of the Jitter Autopoietic System:
- Membrane system for boundary definition and integrity
- Boundary maintenance for production and repair of boundary components
- Configuration validation and integration hooks

The boundary subsystem implements Maturana & Varela's concept of a semipermeable
membrane that defines the autopoietic system while allowing necessary interactions
with the environment.
"""
from .membrane import Membrane
from .boundary_maintenance import BoundaryMaintenance

__all__ = [
    'Membrane',
    'BoundaryMaintenance'
]