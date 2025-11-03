# stephanie/components/jitter/boundary/__init__.py
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
from __future__ import annotations

from .boundary_maintenance import BoundaryMaintenance
from .membrane import Membrane

__all__ = [
    'Membrane',
    'BoundaryMaintenance'
]