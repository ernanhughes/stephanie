# stephanie/components/elm/__init__.py
"""
ELM: Experimental Learning Module for governed self-improvement.
"""

# Core primitives
from .core.context_pack import (
    ContextPack,
    ContextPackCollection,
    ContextType,
    Modality,
    create_user_query_context,
    create_document_context,
    create_embedding_context,
    create_goal_context,
    create_reflection_context,
)
from .core.reward_vector import RewardVector, RewardAxis
from .core.thresholds import (
    CalibratedThresholds,
    create_from_baseline_stats,
    create_conservative_thresholds,
    create_permissive_thresholds,
)

# Tracking & diagnostics
from .tracking.retention_tracker import RetentionTracker, RetentionMetrics
from .tracking.collapse_detector import CollapseDetector, FailureEvent

# Governance layer
from .governance.signal_extractor import GovernanceSignalExtractor
from .governance.dominance_checker import DominanceChecker
from .governance.regime_controller import RegimeController

# Evaluation infrastructure
from .evaluation.governance_reducer import GovernanceReducer, SignalProvider, SignalResult

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

__all__ = [
    # Core
    "ContextPack",
    "ContextPackCollection",
    "ContextType",
    "Modality",
    "RewardVector",
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
    "GovernanceReducer",
    "SignalProvider",
    "SignalResult",
    
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