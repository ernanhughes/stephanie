# evaluation/governance_reducer.py
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Any
from stephanie.components.elm.core.reward_vector import RewardVector
from stephanie.components.elm.core.context_pack import ContextPack
from stephanie.data.score_bundle import ScoreBundle

class SignalProvider(Protocol):
    """Provider interface for governance signals"""
    def compute(
        self,
        context_pack: ContextPack,
        plan_trace: Any,
        output: Any,
        base_bundle: ScoreBundle
    ) -> "SignalResult":
        ...

@dataclass
class SignalResult:
    """Atomic signal contribution from one provider"""
    axis_values: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    failure_signatures: List[str] = field(default_factory=list)
    confidence: float = 1.0

class GovernanceReducer:
    """Pure reducer - aggregates provider signals into governance metrics"""
    
    def __init__(self, providers: List[SignalProvider]):
        self.providers = providers
    
    def evaluate(
        self,
        context_pack: ContextPack,
        plan_trace: Any,
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
                base_bundle=base_bundle
            )
            
            aggregated_values.update(result.axis_values)
            diagnostics.update(result.diagnostics)
            failures.extend(result.failure_signatures)
            confidences.append(result.confidence)

        # Build reward vector
        reward_vector = RewardVector(
            values=aggregated_values,
            trace_id=plan_trace.trace_id,
            source_model="ELM_Governance",
            confidence=sum(confidences) / len(confidences) if confidences else 1.0,
            failure_signatures=failures
        )

        return ScoreBundle(
            reward_vector=reward_vector,
            plan_trace=plan_trace,
            context_pack=context_pack,
            raw_output=output,
            raw_diagnostics=diagnostics
        )