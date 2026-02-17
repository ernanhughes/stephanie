from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from enum import Enum
import logging
import torch
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.plan_trace import PlanTrace
from stephanie.components.elm.core.context_pack import ContextPack
from stephanie.components.elm.core.reward_vector import RewardVector
from stephanie.components.elm.governance.signal_extractor import AxisDirection, RewardAxis
from stephanie.components.elm.evaluation.governance_reducer import SignalProvider

logger = logging.getLogger(__name__)

# ============================================================================
# INTERFACE LAYER — What the evaluator expects from Stephanie's components
# ============================================================================

@runtime_checkable
class HRMInterface(Protocol):
    """Human Reasoning Model alignment scorer"""
    def score_alignment(self, plan_trace: 'PlanTrace', output: Any) -> float:
        """Returns [0.0, 1.0] — higher = more human-aligned"""
        ...

@runtime_checkable
class CertumInterface(Protocol):
    """Hallucination detection and energy projection"""
    def compute_energy(self, output: Any, context_pack: 'ContextPack') -> float:
        """Returns normalized energy [0.0, 1.0] — lower = less hallucinated"""
        ...
    
    def detect_failures(self, energy: float) -> List[str]:
        """Returns failure signatures like ['energy_spike', 'speculative_leap']"""
        ...

@runtime_checkable
class EmbeddingStoreInterface(Protocol):
    """Embedding geometry and margin analysis"""
    def compute_margin(self, input_emb: torch.Tensor, output_emb: torch.Tensor, goal_emb: torch.Tensor) -> float:
        """Returns cosine margin [0.0, 1.0] — higher = better alignment"""
        ...
    
    def compute_metric_stability(self, embeddings: List[torch.Tensor]) -> float:
        """Returns stability score [0.0, 1.0] — higher = more consistent"""
        ...

@runtime_checkable
class PolicyAnalyzerInterface(Protocol):
    """Policy advantage and context fidelity analysis"""
    def compute_advantage(self, plan_trace: 'PlanTrace', baseline_trace: Optional['PlanTrace'] = None) -> float:
        """Returns advantage score [-1.0, 1.0] — higher = better policy choice"""
        ...
    
    def compute_context_fidelity(self, output: Any, context_pack: 'ContextPack') -> float:
        """Returns [0.0, 1.0] — higher = better context grounding"""
        ...

# ============================================================================
# CONFIGURATION — What makes this evaluator adaptive
# ============================================================================

class EvaluationMode(str, Enum):
    """Predefined evaluation profiles"""
    WIKIPEDIA = "wikipedia"        # Strict on energy, HRM
    RESEARCH = "research"          # Strict on margin, metric stability
    EXPLORATION = "exploration"    # Loose thresholds, high creativity tolerance
    AGGRESSIVE = "aggressive"      # Maximize improvement velocity
    CONSERVATIVE = "conservative"  # Prioritize stability over novelty

@dataclass
class AxisConfig:
    """Per-axis evaluation configuration"""
    enabled: bool = True
    weight: float = 1.0
    threshold: Optional[float] = None  # For reflection triggers
    direction: 'AxisDirection' = AxisDirection.HIGHER_IS_BETTER

@dataclass
class MultiLayerEvaluatorConfig:
    """Full evaluator configuration"""
    mode: EvaluationMode = EvaluationMode.CONSERVATIVE
    
    # Per-axis configs
    axes: Dict[RewardAxis, AxisConfig] = field(default_factory=dict)
    
    # Component toggles
    use_hrm: bool = True
    use_certum: bool = True
    use_embedding_store: bool = True
    use_policy_analyzer: bool = True
    
    # Normalization
    normalize_to_unit_sphere: bool = True
    
    def __post_init__(self):
        # Default axis configs by mode
        if not self.axes:
            defaults = {
                EvaluationMode.WIKIPEDIA: {
                    RewardAxis.HRM_ALIGNMENT: AxisConfig(weight=1.5, threshold=0.7),
                    RewardAxis.HALLUCINATION_ENERGY: AxisConfig(weight=2.0, threshold=0.3, direction=AxisDirection.LOWER_IS_BETTER),
                    RewardAxis.EMBEDDING_MARGIN: AxisConfig(weight=1.0),
                },
                EvaluationMode.RESEARCH: {
                    RewardAxis.EMBEDDING_MARGIN: AxisConfig(weight=1.5, threshold=0.5),
                    RewardAxis.METRIC_ALIGNMENT: AxisConfig(weight=1.5, threshold=0.6),
                    RewardAxis.HRM_ALIGNMENT: AxisConfig(weight=1.0),
                },
                EvaluationMode.EXPLORATION: {
                    RewardAxis.POLICY_ADVANTAGE: AxisConfig(weight=1.2),
                    RewardAxis.COHERENCE: AxisConfig(weight=1.0),
                    # Higher energy tolerance
                    RewardAxis.HALLUCINATION_ENERGY: AxisConfig(weight=0.5, threshold=0.6, direction=AxisDirection.LOWER_IS_BETTER),
                },
                EvaluationMode.CONSERVATIVE: {
                    RewardAxis.HRM_ALIGNMENT: AxisConfig(weight=1.0, threshold=0.6),
                    RewardAxis.CONTEXT_FIDELITY: AxisConfig(weight=1.0, threshold=0.6),
                    RewardAxis.HALLUCINATION_ENERGY: AxisConfig(weight=1.0, threshold=0.4, direction=AxisDirection.LOWER_IS_BETTER),
                }
            }
            self.axes = defaults[self.mode]

# ============================================================================
# MAIN EVALUATOR — The integration engine
# ============================================================================

class GovernanceReducer:
    def __init__(self, providers: List[SignalProvider]):
        self.providers = providers

    def evaluate(
        self,
        context_pack: ContextPack,
        plan_trace: PlanTrace,
        output: Any,
        base_bundle: ScoreBundle
    ) -> ScoreBundle:

        aggregated_values = {}
        diagnostics = {}
        failures = []
        confidences = []

        for provider in self.providers:
            result = provider.compute(
                context_pack=context_pack,
                plan_trace=plan_trace,
                output=output,
                score_bundle=base_bundle
            )

            aggregated_values.update(result.axis_values)
            diagnostics.update(result.diagnostics)
            failures.extend(result.failure_signatures)
            confidences.append(result.confidence)

        reward_vector = RewardVector(
            values=aggregated_values,
            trace_id=plan_trace.trace_id,
            source_model="ELM_Governance",
            confidence=sum(confidences)/len(confidences) if confidences else 1.0,
            failure_signatures=failures
        )

        return ScoreBundle(
            reward_vector=reward_vector,
            plan_trace=plan_trace,
            context_pack=context_pack,
            raw_output=output,
            raw_diagnostics=diagnostics
        )
