# stephanie/components/jitter/regulation/reproduction/__init__.py
"""
reproduction/__init__.py
=======================
Reproduction subsystem initialization module.

This module initializes the reproduction components of the Jitter Autopoietic System:
- Core reproduction logic
- Quality control for offspring
- Heritage preservation for genetic continuity
- Configuration validation and integration hooks

The reproduction subsystem implements the autopoietic system's ability to create
new organisms while maintaining genetic diversity and quality standards.
"""
from __future__ import annotations

from .heritage_manager import HeritageManager
from .quality_control import QualityControlledReproduction
from .reproduction_system import ReproductionSystem

__all__ = [
    'ReproductionSystem',
    'QualityControlledReproduction',
    'HeritageManager'
]