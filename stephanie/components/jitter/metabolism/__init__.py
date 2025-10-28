"""
metabolism/__init__.py
======================
Metabolism subsystem initialization module.

This module initializes the metabolism components of the Jitter Autopoietic System:
- Energy pools for cognitive, metabolic, and reserve energy
- Metabolic pathways for energy conversion
- Energy optimization for performance and efficiency
- Configuration validation and integration hooks

The metabolism subsystem implements the energy management necessary for
maintaining the autopoietic system's organizationally closed production.
"""
from .energy import EnergyPools
from .metabolic_pathways import MetabolicPathways
from .energy_optimizer import EnergyOptimizer

__all__ = [
    'EnergyPools',
    'MetabolicPathways',
    'EnergyOptimizer'
]