# stephanie/components/jitter/lifecycle/__init__.py
"""
lifecycle/__init__.py
====================
Lifecycle subsystem initialization module.

This module initializes the lifecycle components of the Jitter Autopoietic System:
- Orchestrator for pipeline integration
- Main execution agent for JAS lifecycle
- SSP integration hooks
- Reward shaping for learning
- Configuration validation and integration hooks

The lifecycle subsystem implements the complete operational cycle of Jitter
from initialization through execution to termination.
"""
from __future__ import annotations

from .integration.reward_shaper import RewardShaper
from .integration.ssp_integration import SSPIntegration
from .lifecycle_agent import JASLifecycleAgent
from .orchestrator import JASOrchestrator

__all__ = [
    'JASOrchestrator',
    'JASLifecycleAgent',
    'SSPIntegration',
    'RewardShaper'
]