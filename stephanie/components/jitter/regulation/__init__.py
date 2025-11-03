# stephanie/components/jitter/regulation/__init__.py
"""
regulation/__init__.py
====================
Regulation subsystem initialization module.

This module initializes the regulation components of the Jitter Autopoietic System:
- Homeostasis controllers for physiological regulation
- Apoptosis system for programmed cell death
- Reproduction system for self-reproduction
- Configuration validation and integration hooks

The regulation subsystem implements the autopoietic system's ability to maintain
internal stability while adapting to environmental changes and preparing for
reproduction or death.
"""
from __future__ import annotations

from .apoptosis import ApoptosisSystem
from .homeostasis.adaptive_setpoints import AdaptiveSetpoints
from .homeostasis.controller import PIDController
from .homeostasis.crisis_detector import CrisisDetector
from .reproduction.heritage_manager import HeritageManager
from .reproduction.quality_control import QualityControlledReproduction
from .reproduction.reproduction_system import ReproductionSystem

__all__ = [
    'PIDController',
    'AdaptiveSetpoints',
    'CrisisDetector',
    'ApoptosisSystem',
    'ReproductionSystem',
    'QualityControlledReproduction',
    'HeritageManager'
]