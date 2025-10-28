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
from .homeostasis.controller import PIDController
from .homeostasis.adaptive_setpoints import AdaptiveSetpoints
from .homeostasis.crisis_detector import CrisisDetector
from .apoptosis import ApoptosisSystem
from .reproduction.reproduction_system import ReproductionSystem
from .reproduction.quality_control import QualityControlledReproduction
from .reproduction.heritage_manager import HeritageManager

__all__ = [
    'PIDController',
    'AdaptiveSetpoints',
    'CrisisDetector',
    'ApoptosisSystem',
    'ReproductionSystem',
    'QualityControlledReproduction',
    'HeritageManager'
]