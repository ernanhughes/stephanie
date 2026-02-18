# stephanie/components/elm/__init__.py
"""
ELM: Experimental Learning Module for governed self-improvement.
"""

# Core primitives
from stephanie.components.elm.orchestration.system_interface import SystemInterface
from .core.context_pack import (
    ContextPack,
    ContextPackCollection,
    ContextType,
    Modality,
)
from .core.thresholds import (
    CalibratedThresholds,
)

# Tracking & diagnostics
from .tracking.retention_tracker import RetentionTracker, RetentionMetrics
from .tracking.collapse_detector import CollapseDetector, FailureEvent

# Governance layer
from .governance.signal_extractor import GovernanceSignalExtractor
from .governance.dominance_checker import DominanceChecker
from .governance.regime_controller import RegimeController

# Evaluation infrastructure
from .evaluation.governance_scorer import GovernanceScorer

# Experiment harness
from .experiment.baseline_calibrator import BaselineCalibrator
from .experiment.experiment_harness import ScoreBundleExperiment
from .experiment.experiment_persistence import ExperimentPersistence
from .experiment.perturbation_injector import (
    PerturbationInjector,
    PerturbationConfig,
    create_perturbation_config,
    register_custom_severity
)
from .orchestration.orchestrator import ELMOrchestrator
from .orchestration.system_interface import  SystemInterface

__all__ = [
    # Core
    "ContextPack",
    "ContextPackCollection",
    "ContextType",
    "Modality",
    "RewardAxis",
    "CalibratedThresholds",
    
    # Tracking
    "RetentionTracker",
    "RetentionMetrics",
    "CollapseDetector",
    "FailureEvent",
    
    # Governance
    "GovernanceSignalExtractor",
    "DominanceChecker",
    "RegimeController",
    
    # Evaluation
    "GovernanceScorer",

    # Experiment
    "BaselineCalibrator",
    "ScoreBundleExperiment",
    "ExperimentPersistence",
    "PerturbationInjector",
    "PerturbationInjector",
    "PerturbationConfig",
    "create_perturbation_config",
    "register_custom_severity",
    
    # Orchestration
    "ELMOrchestrator",
    "SystemInterface",
]