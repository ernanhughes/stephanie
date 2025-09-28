# stephanie/experiments/ab_engine.py
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

T = TypeVar('T')  # Generic type for experiment variants

_logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment"""
    name: str
    min_samples_per_group: int = 8
    window_seconds: Optional[int] = None
    max_p_value: float = 0.10
    min_effect_size: float = 0.147
    ab_cooldown_sec: float = 1800
    rng_seed: Optional[int] = None
    domain: Optional[str] = None

@dataclass
class ExperimentResult:
    """Result of an A/B experiment validation"""
    samples_A: int
    samples_B: int
    mean_A: float
    mean_B: float
    delta_B_minus_A: float
    relative_improvement: float
    p_value: float
    recommend_commit: bool
    decision_reason: str
    timestamp: float = field(default_factory=time.time)

class ABEngine(Generic[T]):
    """
    Reusable A/B testing engine that can be applied to any metric-based experiment.
    
    Usage:
        # 1. Initialize with config
        ab = ABEngine(ExperimentConfig(name="verification_strategy", min_samples_per_group=10))
        
        # 2. Enroll in experiment
        variant = ab.enroll(context={"case_id": 123}, variant_factory=lambda: current_strategy)
        
        # 3. Record results
        ab.record_result("A", 0.85, context={"case_id": 123, "domain": "transformers"})
        
        # 4. Validate periodically
        result = ab.validate()
        if result and result.recommend_commit:
            # Apply the winning variant
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = random.Random(config.rng_seed) if config.rng_seed is not None else random.Random()
        self._ab_buffer: List[Dict[str, Any]] = []  # rolling buffer of recent results
        self._last_commit_ts: float = 0.0
        self._name = config.name
    
    def enroll(self, context: Dict[str, Any], variant_factory: Callable[[], T]) -> T:
        """Enroll in experiment and get variant to test"""
        case_id = context.get("case_id")
        if case_id is not None and self.config.rng_seed is None:
            # Deterministic assignment based on case_id
            return variant_factory() if (hash((case_id, self._name)) & 1) else variant_factory()
        return variant_factory() if self.rng.random() < 0.5 else variant_factory()
    
    def record_result(self, group: str, performance: float, context: Dict[str, Any]) -> None:
        """Record result of an experiment trial"""
        if group not in ("A", "B"):
            _logger.warning(f"Invalid group '{group}' for experiment '{self._name}'")
            return
            
        sample = {
            "group": group,
            "performance": performance,
            "domain": context.get("domain", "default"),
            "ts": time.time(),
            "case_id": context.get("case_id"),
        }
        
        # Optional efficiency metrics
        for k in ("tokens", "cost", "wall_sec"):
            if k in context:
                sample[k] = context[k]
                
        self._ab_buffer.append(sample)
        if len(self._ab_buffer) > 400:  # Keep buffer size reasonable
            self._ab_buffer = self._ab_buffer[-400:]
    
    def validate(self) -> Optional[ExperimentResult]:
        """Validate which variant is better based on collected data"""
        now = time.time()
        samples = self._ab_buffer.copy()
        
        # Apply time window filter if configured
        if self.config.window_seconds is not None:
            samples = [s for s in samples if (now - s["ts"]) <= self.config.window_seconds]
        
        # Split into groups
        a_samples = [s for s in samples if s["group"] == "A"]
        b_samples = [s for s in samples if s["group"] == "B"]
        
        if len(a_samples) < self.config.min_samples_per_group or len(b_samples) < self.config.min_samples_per_group:
            return None
        
        # Calculate statistics
        perf_a = [s["performance"] for s in a_samples]
        perf_b = [s["performance"] for s in b_samples]
        
        mean_a = sum(perf_a) / len(perf_a)
        mean_b = sum(perf_b) / len(perf_b)
        delta = mean_b - mean_a
        rel_impr = delta / abs(mean_a) if mean_a != 0 else 0.0
        
        # Simple t-test approximation
        # (In production, you'd want a proper statistical test)
        n_a, n_b = len(a_samples), len(b_samples)
        pooled_std = ((n_a-1)*self._std(perf_a)**2 + (n_b-1)*self._std(perf_b)**2) / (n_a + n_b - 2)
        pooled_std = pooled_std**0.5
        se = pooled_std * (1/n_a + 1/n_b)**0.5
        t = delta / se if se > 0 else 0
        p_value = self._t_to_p(t, n_a + n_b - 2)
        
        # Decision logic
        recommend_commit = (
            delta > 0 and 
            p_value <= self.config.max_p_value and 
            rel_impr >= 0.02  # Minimum 2% improvement
        )
        
        reason = "ok" if recommend_commit else "insufficient_evidence"
        
        return ExperimentResult(
            samples_A=len(a_samples),
            samples_B=len(b_samples),
            mean_A=mean_a,
            mean_B=mean_b,
            delta_B_minus_A=delta,
            relative_improvement=rel_impr,
            p_value=p_value,
            recommend_commit=recommend_commit,
            decision_reason=reason
        )
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _t_to_p(self, t: float, df: int) -> float:
        """Approximate p-value from t-statistic (simplified)"""
        # This is a simplified approximation - in production use proper stats library
        return min(1.0, abs(t) * 0.1)  # Simplified for example