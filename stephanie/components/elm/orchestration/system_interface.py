"""
SystemInterface: Contract definition for Stephanie engine integration with ELM experimental harness.

This protocol defines the minimal interface required for any system to participate in
governed self-improvement experiments. Implementations must satisfy all methods.

Design principles:
- Minimal surface area (only experiment-required methods)
- Type-safe with clear contracts
- Compatible with ScoreBundle persistence layer
- No internal system details exposed
- Testable via mock implementations
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable
from stephanie.data.score_bundle import ScoreBundle

@runtime_checkable
class SystemInterface(Protocol):
    """
    Protocol defining required methods for ELM experimental integration.
    
    Any Stephanie engine variant must implement this interface to participate
    in calibrated self-improvement experiments.
    
    Usage:
        class MyStephanieEngine(SystemInterface):
            def evaluate(self, query: Any) -> ScoreBundle:
                # Implementation
                pass
            
            def attempt_improvement(
                self, 
                query: Any, 
                bundle_before: ScoreBundle
            ) -> Optional[Dict[str, Any]]:
                # Implementation
                pass
            
            def commit_improvement(
                self,
                query: Any,
                bundle_after: ScoreBundle,
                reflection_trace: Optional[Any] = None
            ) -> None:
                # Implementation
                pass
        
        # Validate implementation
        assert isinstance(my_engine, SystemInterface)
    """
    
    def evaluate(self, query: Any) -> ScoreBundle:
        """
        Evaluate query and return ScoreBundle.
        
        Must:
        - Return fully populated ScoreBundle with all critical dimensions
        - Include raw_energy in attributes for energy extraction
        - Be deterministic for identical queries (for reproducibility)
        
        Args:
            query: Input query (type system-specific)
            
        Returns:
            ScoreBundle containing evaluation results
            
        Raises:
            EvaluationError: If evaluation fails catastrophically
        """
        ...
    
    def attempt_improvement(
        self,
        query: Any,
        bundle_before: ScoreBundle
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt self-improvement via reflection/retry cycle.
        
        Must:
        - Return None if no improvement attempt made
        - Return dict with "bundle" key containing improved ScoreBundle
        - Optionally include "reflection" key with trace metadata
        - Only return bundle if dominance check would pass (pre-filter)
        
        Args:
            query: Original query
            bundle_before: ScoreBundle from initial evaluation
            
        Returns:
            Dict with keys:
                - "bundle": Improved ScoreBundle (required)
                - "reflection": Reflection trace metadata (optional)
            None if no improvement attempted
            
        Raises:
            ImprovementError: If improvement process fails
        """
        ...
    
    def commit_improvement(
        self,
        query: Any,
        bundle_after: ScoreBundle,
        reflection_trace: Optional[Any] = None
    ) -> None:
        """
        Persist validated improvement to system state.
        
        Must:
        - Update internal model state with improvement
        - Persist to database via existing ScoreBundle.save_bundle()
        - Include reflection_trace in metadata if provided
        - Be idempotent (safe to call multiple times)
        
        Args:
            query: Original query context
            bundle_after: Validated improved ScoreBundle
            reflection_trace: Optional reflection metadata for provenance
            
        Raises:
            CommitError: If persistence fails
        """
        ...
    
    def inject_misleading_evidence(self, probability: float = 0.4) -> None:
        """
        Inject controlled perturbation for stress testing.
        
        Used by PerturbationInjector during experiment.
        Must be reversible via restore_original_state().
        
        Args:
            probability: Likelihood of injecting misleading evidence [0.0, 1.0]
        """
        ...
    
    def increase_query_complexity(self, factor: float = 1.5) -> None:
        """
        Increase query complexity for stress testing.
        
        Used by PerturbationInjector during experiment.
        Must be reversible via restore_original_state().
        
        Args:
            factor: Complexity multiplier (>1.0 increases complexity)
        """
        ...
    
    def restore_original_state(self) -> None:
        """
        Restore system to pre-perturbation state.
        
        Must reverse all effects of:
        - inject_misleading_evidence()
        - increase_query_complexity()
        - temporarily_disable_governance()
        
        Called after perturbation episode completes.
        """
        ...
    
    def get_query_id(self, query: Any) -> Optional[str]:
        """
        Extract stable identifier from query.
        
        Used for experiment logging and reproducibility.
        Must return consistent ID for identical queries.
        
        Args:
            query: Input query
            
        Returns:
            String identifier or None if not available
        """
        ...
    
    @property
    def name(self) -> str:
        """System identifier for experiment logging (e.g., 'stephanie_v3_governed')"""
        ...
    
    @property
    def version(self) -> str:
        """Semantic version string (e.g., '2.1.0')"""
        ...


# ============================================================================
# ERROR TYPES FOR INTERFACE CONTRACT
# ============================================================================

class EvaluationError(Exception):
    """Raised when query evaluation fails catastrophically"""
    pass

class ImprovementError(Exception):
    """Raised when improvement attempt fails"""
    pass

class CommitError(Exception):
    """Raised when improvement persistence fails"""
    pass


# ============================================================================
# MOCK IMPLEMENTATION FOR TESTING
# ============================================================================

class MockSystem(SystemInterface):
    """
    Minimal mock implementation for testing experiment harness.
    
    Usage:
        mock = MockSystem()
        experiment = ScoreBundleExperiment(system=mock, ...)
        result = experiment.run()  # Validates harness logic
    """
    
    def __init__(self, seed: int = 42):
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self._original_state = {}
        self._perturbed = False
    
    def evaluate(self, query: Any) -> ScoreBundle:
        from stephanie.data.score_result import ScoreResult
        
        # Simulate realistic scores with controlled variance
        base_energy = 0.25 + self.rng.normal(0, 0.05)
        base_hrm = 0.82 + self.rng.normal(0, 0.04)
        base_margin = 0.65 + self.rng.normal(0, 0.06)
        
        results = {
            "alignment": ScoreResult(
                dimension="alignment",
                score=base_hrm * 100,
                source="mock_hrm",
                rationale="Simulated HRM evaluation",
                weight=1.0,
                attributes={"raw_energy": base_energy * 100}
            ),
            "energy": ScoreResult(
                dimension="energy",
                score=(1.0 - base_energy) * 100,  # Inverted for lower=better
                source="mock_ebt",
                rationale="Simulated energy evaluation",
                weight=1.0,
                attributes={"raw_energy": base_energy * 100}
            ),
            "margin": ScoreResult(
                dimension="margin",
                score=base_margin * 100,
                source="mock_embedding",
                rationale="Simulated margin evaluation",
                weight=1.0,
                attributes={"embedding_margin": base_margin}
            )
        }
        
        return ScoreBundle(results=results)
    
    def attempt_improvement(
        self,
        query: Any,
        bundle_before: ScoreBundle
    ) -> Optional[Dict[str, Any]]:
        # Simulate 70% improvement success rate
        if self.rng.random() < 0.7:
            improved_bundle = self.evaluate(query)
            return {"bundle": improved_bundle, "reflection": {"applied": True}}
        return None
    
    def commit_improvement(
        self,
        query: Any,
        bundle_after: ScoreBundle,
        reflection_trace: Optional[Any] = None
    ) -> None:
        # No-op for mock (persistence simulated)
        pass
    
    def inject_misleading_evidence(self, probability: float = 0.4) -> None:
        self._perturbed = True
    
    def increase_query_complexity(self, factor: float = 1.5) -> None:
        self._perturbed = True
    
    def restore_original_state(self) -> None:
        self._perturbed = False
    
    def get_query_id(self, query: Any) -> Optional[str]:
        return str(hash(str(query)))[:16]
    
    @property
    def name(self) -> str:
        return "mock_stephanie_test_system"
    
    @property
    def version(self) -> str:
        return "test-1.0"


# ============================================================================
# VALIDATION UTILITY
# ============================================================================

def validate_system_implementation(system: Any) -> bool:
    """
    Validate that system satisfies SystemInterface contract.
    
    Usage:
        assert validate_system_implementation(my_engine)
    
    Checks:
    - All required methods exist
    - Properties are accessible
    - Minimal type compatibility
    
    Returns:
        True if valid, raises AssertionError otherwise
    """
    import inspect
    
    # Check protocol compliance
    assert isinstance(system, SystemInterface), \
        "System must implement SystemInterface protocol"
    
    # Check required methods exist
    required_methods = [
        'evaluate', 'attempt_improvement', 'commit_improvement',
        'inject_misleading_evidence', 'increase_query_complexity',
        'restore_original_state', 'get_query_id'
    ]
    for method in required_methods:
        assert hasattr(system, method), \
            f"System missing required method: {method}"
        assert callable(getattr(system, method)), \
            f"System.{method} must be callable"
    
    # Check required properties exist
    required_props = ['name', 'version']
    for prop in required_props:
        assert hasattr(system, prop), \
            f"System missing required property: {prop}"
    
    # Check method signatures (basic)
    sig = inspect.signature(system.evaluate)
    assert 'query' in sig.parameters, \
        "evaluate() must accept 'query' parameter"
    
    sig = inspect.signature(system.attempt_improvement)
    assert 'query' in sig.parameters and 'bundle_before' in sig.parameters, \
        "attempt_improvement() must accept 'query' and 'bundle_before'"
    
    sig = inspect.signature(system.commit_improvement)
    assert 'query' in sig.parameters and 'bundle_after' in sig.parameters, \
        "commit_improvement() must accept 'query' and 'bundle_after'"
    
    return True


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "SystemInterface",
    "EvaluationError",
    "ImprovementError",
    "CommitError",
    "MockSystem",
    "validate_system_implementation"
]