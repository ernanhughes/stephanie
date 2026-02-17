from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Any


@dataclass
class SignalResult:
    """Atomic signal contribution from one provider"""
    dimensions: Dict[str, Dict[str, Any]]  # ✅ Matches GovernanceProvider output
    # Example: {"hallucination_energy": {"score": 0.25, "rationale": "...", "attributes": {...}}}
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    failure_signatures: List[str] = field(default_factory=list)
    confidence: float = 1.0


class SignalProvider(Protocol):
    def compute(
        self,
        context_pack: Any,
        plan_trace: Any,
        output: Any,
        **kwargs
    ) -> SignalResult:
        ...
