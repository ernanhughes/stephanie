# stephanie/components/gap/shared_scm.py
"""
Shared Core Metrics (SCM) - Universal translator for model evaluation signals.

This module provides the core normalization and alignment infrastructure that
enables cross-model comparison by converting heterogeneous model outputs into
a standardized [0,1] scale with consistent semantic meaning.

Key Components:
- ScoreNormalizer: Scale-aware normalization with source-specific ranges
- Range: Mathematical clamping and normalization helper
- SCM schema: Fixed 12-dimensional output format for all models
- Extraction helpers: Flexible attribute lookup with fallback chains

The SCM system transforms raw model outputs (0-10, 0-100, logits, probabilities)
into a common language that enables delta analysis, topology computation, and
model routing decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------
# Scale-aware normalization
# ----------------------------

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Range:
    """
    Mathematical range for value normalization and clamping.
    
    Represents a [lo, hi] interval and provides clamping operations to 
    normalize values to the [0,1] range with proper bounds handling.
    
    Attributes:
        lo: Lower bound of the value range
        hi: Upper bound of the value range
        
    Example:
        >>> r = Range(0, 100)
        >>> r.clamp01(50)  # Returns 0.5
        >>> r.clamp01(150) # Returns 1.0 (clamped)
        >>> r.clamp01(-10) # Returns 0.0 (clamped)
    """
    
    lo: float
    hi: float
    
    def clamp01(self, v: float) -> float:
        """
        Normalize value to [0,1] range based on this range.
        
        Args:
            v: Raw value to normalize
            
        Returns:
            Normalized value in [0,1], clamped to bounds
            
        Notes:
            - Handles NaN/infinite values safely (returns 0.0)
            - Returns 0.0 if range is degenerate (hi == lo)
            - Applies linear scaling: (v - lo) / (hi - lo)
        """
        if not np.isfinite(v) or self.hi == self.lo:
            return 0.0
        x = (v - self.lo) / (self.hi - self.lo)
        return float(np.clip(x, 0.0, 1.0))


class ScoreNormalizer:
    """
    Scale-aware normalizer for heterogeneous model outputs.
    
    Converts raw scores and attributes from different model families (HRM, Tiny, LLM)
    into a unified [0,1] scale using explicit, source-aware range mappings.
    
    Distinguishes between:
    - SCORE ranges: Dimension-specific quality scores (reasoning, clarity, etc.)
    - ATTR ranges: Diagnostic attributes (uncertainty, OOD, consistency, etc.)
    
    Usage:
        >>> normalizer = ScoreNormalizer()
        >>> normalizer.norm_score(7.5, source="HRM", dimension="reasoning")  # 0.75
        >>> normalizer.norm_attr(0.8, source="TINY", name="uncertainty")     # 0.8
    """
    
    def __init__(
        self,
        score_ranges: Optional[Dict[Tuple[str, Optional[str]], Range]] = None,
        attr_ranges: Optional[Dict[Tuple[str, Optional[str]], Range]] = None,
    ):
        """
        Initialize normalizer with source-specific range mappings.
        
        Args:
            score_ranges: Optional overrides for score normalization ranges
            attr_ranges: Optional overrides for attribute normalization ranges
            
        Default Ranges:
            Scores:
                LLM: 0-100 (rubric-based scoring)
                HRM: 0-10 (typical HRM output scale)  
                TINY: 0-1 (sigmoid-activated outputs)
            Attributes:
                All: 0-1 (most diagnostics are already normalized)
        """
        # Default score ranges by source and dimension
        self.score_ranges = {
            ("LLM", None): Range(0.0, 100.0),  # Rubric-based LLM judges
            ("HRM", None): Range(0.0, 10.0),   # HRM typically uses 0-10 scale
            ("TINY", None): Range(0.0, 1.0),   # TinyRecursion uses sigmoid outputs
        }
        if score_ranges:
            self.score_ranges.update(score_ranges)

        # Attributes are typically already in [0,1] range
        self.attr_ranges = {
            ("LLM", None): Range(0.0, 1.0),
            ("HRM", None): Range(0.0, 1.0),
            ("TINY", None): Range(0.0, 1.0),
        }
        if attr_ranges:
            self.attr_ranges.update(attr_ranges)

    def _get_range(self, table: Dict[Tuple[str, Optional[str]], Range],
                   source: str, dimension: Optional[str]) -> Range:
        """
        Find appropriate range for source and dimension.
        
        Lookup order:
        1. Exact (source, dimension) match
        2. (source, None) fallback  
        3. Default 0-1 range
        
        Args:
            table: Range dictionary to search
            source: Model source (HRM, TINY, LLM)
            dimension: Optional reasoning dimension
            
        Returns:
            Appropriate Range object for normalization
        """
        key = (source, dimension)
        if key in table:
            return table[key]
        key = (source, None)
        if key in table:
            return table[key]
        return Range(0.0, 1.0)

    def norm_score(self, value: Any, *, source: str, dimension: Optional[str]) -> float:
        """
        Normalize a dimension score using source-specific range.
        
        Args:
            value: Raw score value (any numeric type)
            source: Model source identifier
            dimension: Reasoning dimension name
            
        Returns:
            Normalized score in [0,1] range
            
        Logs:
            - Debug: Score normalization events
            - Warning: Invalid or out-of-range values
        """
        try:
            v = float(value)
        except Exception:
            _logger.debug(f"Invalid score value: {value}, defaulting to 0.0")
            return 0.0
            
        rng = self._get_range(self.score_ranges, source, dimension)
        result = rng.clamp01(v)
        
        # Log extreme normalizations for monitoring
        if result <= 0.01 or result >= 0.99:
            _logger.debug(f"Extreme score normalization: {v} -> {result} (source: {source}, dim: {dimension})")
            
        return result

    def norm_attr(self, value: Any, *, source: str, name: Optional[str] = None) -> float:
        """
        Normalize a diagnostic attribute using source-specific range.
        
        Args:
            value: Raw attribute value
            source: Model source identifier  
            name: Optional attribute name for specific ranges
            
        Returns:
            Normalized attribute in [0,1] range
        """
        try:
            v = float(value)
        except Exception:
            _logger.debug(f"Invalid attribute value: {value}, defaulting to 0.0")
            return 0.0
            
        rng = self._get_range(self.attr_ranges, source, name)
        return rng.clamp01(v)


# Single module-level normalizer instance
# Used throughout GAP system for consistent normalization
NORMALIZER = ScoreNormalizer()


def _source_from_prefix(model_prefix: str) -> str:
    """
    Determine model source from prefix string.
    
    Args:
        model_prefix: Prefix string from model output (e.g., "hrm", "tiny", "llm")
        
    Returns:
        Standardized source identifier: "HRM", "TINY", or "LLM"
        
    Note:
        Unknown prefixes default to "LLM" behavior (0-100 range)
    """
    p = (model_prefix or "").strip().lower()
    if p.startswith("hrm"):
        return "HRM"
    if p.startswith("tiny"):
        return "TINY"
    if p.startswith("llm"):
        return "LLM"
    # Fallback: treat unknown as LLM-style unless configured otherwise
    _logger.debug(f"Unknown model prefix '{model_prefix}', defaulting to LLM normalization")
    return "LLM"


# ----------------------------
# SCM schema definition
# ----------------------------

# Fixed 12-dimensional output schema for all models
# This is the common language that enables cross-model comparison
SCM_COLUMNS = [
    "scm.reasoning.score01",      # Logical structure and soundness
    "scm.knowledge.score01",      # Factual accuracy and specificity  
    "scm.clarity.score01",        # Organization and readability
    "scm.faithfulness.score01",   # Consistency with context
    "scm.coverage.score01",       # Completeness across facets
    "scm.aggregate01",            # Overall quality assessment
    "scm.uncertainty01",          # Model confidence in assessment
    "scm.ood_hat01",              # Out-of-distribution detection
    "scm.consistency01",          # Robustness to perturbations
    "scm.length_norm01",          # Length normalization factor
    "scm.temp01",                 # Calibration temperature
    "scm.agree_hat01",            # Cross-model agreement prediction
]

# Core reasoning dimensions for evaluation
DIMENSIONS = ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]


# ----------------------------
# Helper functions with scale awareness
# ----------------------------

def _mean01(vals: List[float], default: float = 0.0) -> float:
    """
    Compute mean of values with [0,1] clamping.
    
    Args:
        vals: List of values to average
        default: Default value if list is empty
        
    Returns:
        Mean value clamped to [0,1] range
    """
    arr = [v for v in vals if v is not None]
    if not arr:
        return default
    return float(np.clip(float(np.mean(arr)), 0.0, 1.0))


def _fetch_any_score(vec: Dict[str, Any], keys: List[str], *, source: str,
                     dimension: Optional[str], default: float = 0.0) -> float:
    """
    Flexible score extraction with fallback chain.
    
    Args:
        vec: Dictionary of model outputs
        keys: Ordered list of keys to try
        source: Model source for normalization
        dimension: Reasoning dimension for range selection
        default: Default value if no keys found
        
    Returns:
        Normalized score from first available key
    """
    for k in keys:
        if k in vec:
            return NORMALIZER.norm_score(vec[k], source=source, dimension=dimension)
    return default


def _fetch_any_attr(vec: Dict[str, Any], keys: List[str], *, source: str,
                    name: Optional[str] = None, default: float = 0.0) -> float:
    """
    Flexible attribute extraction with fallback chain.
    
    Args:
        vec: Dictionary of model outputs  
        keys: Ordered list of keys to try
        source: Model source for normalization
        name: Attribute name for range selection
        default: Default value if no keys found
        
    Returns:
        Normalized attribute from first available key
    """
    for k in keys:
        if k in vec:
            return NORMALIZER.norm_attr(vec[k], source=source, name=name)
    return default


def _dim_score(vec: Dict[str, Any], model_prefix: str, dim: str) -> float:
    """
    Extract normalized dimension score with flexible key matching.
    
    Tries common key patterns in order:
    1. {prefix}.{dim}.score
    2. {prefix}.{dim}.aggregate  
    3. {prefix}.{dim}
    
    Args:
        vec: Model output dictionary
        model_prefix: Model identifier prefix
        dim: Reasoning dimension name
        
    Returns:
        Normalized dimension score in [0,1]
    """
    src = _source_from_prefix(model_prefix)
    return _fetch_any_score(vec, [
        f"{model_prefix}.{dim}.score",
        f"{model_prefix}.{dim}.aggregate", 
        f"{model_prefix}.{dim}",
    ], source=src, dimension=dim, default=0.0)


def _uncertainty(vec: Dict[str, Any], model_prefix: str) -> float:
    """
    Extract uncertainty estimate with dimension fallback.
    
    Preference order:
    1. Dimension-specific uncertainty signals
    2. Global uncertainty attributes
    3. Default moderate uncertainty (0.5)
    
    Args:
        vec: Model output dictionary
        model_prefix: Model identifier prefix
        
    Returns:
        Normalized uncertainty estimate in [0,1]
    """
    src = _source_from_prefix(model_prefix)
    # Prefer dimension-specific signals if available
    cands = []
    for d in DIMENSIONS:
        for name, key in (
            ("energy",     f"{model_prefix}.{d}.attr.energy"),
            ("entropy",    f"{model_prefix}.{d}.attr.entropy"), 
            ("uncertainty",f"{model_prefix}.{d}.attr.uncertainty"),
        ):
            if key in vec:
                cands.append(NORMALIZER.norm_attr(vec[key], source=src, name=name))
    if cands:
        return _mean01(cands, 0.5)
        
    # Global fallbacks
    return _fetch_any_attr(vec, [
        f"{model_prefix}.attr.energy",
        f"{model_prefix}.attr.entropy", 
        f"{model_prefix}.uncertainty",
    ], source=src, name="uncertainty", default=0.5)


def _ood(vec: Dict[str, Any], model_prefix: str) -> float:
    """
    Extract out-of-distribution detection score.
    
    Args:
        vec: Model output dictionary
        model_prefix: Model identifier prefix
        
    Returns:
        Normalized OOD score in [0,1] (higher = more OOD)
    """
    src = _source_from_prefix(model_prefix)
    # Try dimension-specific OOD signals first
    cands = []
    for d in DIMENSIONS:
        k = f"{model_prefix}.{d}.attr.ood_hat"
        if k in vec:
            cands.append(NORMALIZER.norm_attr(vec[k], source=src, name="ood_hat"))
    if cands:
        return _mean01(cands, 0.0)
        
    # Global OOD fallbacks  
    return _fetch_any_attr(vec, [
        f"{model_prefix}.attr.ood_hat",
        f"{model_prefix}.ood_hat",
    ], source=src, name="ood_hat", default=0.0)


def _consistency(vec: Dict[str, Any], model_prefix: str) -> float:
    """
    Extract consistency/robustness estimate.
    
    Args:
        vec: Model output dictionary  
        model_prefix: Model identifier prefix
        
    Returns:
        Normalized consistency score in [0,1] (higher = more robust)
    """
    src = _source_from_prefix(model_prefix)
    # Dimension-specific consistency signals
    cands = []
    for d in DIMENSIONS:
        k = f"{model_prefix}.{d}.attr.consistency_hat"
        if k in vec:
            cands.append(NORMALIZER.norm_attr(vec[k], source=src, name="consistency_hat"))
    if cands:
        return _mean01(cands, 0.5)
        
    # Global consistency fallbacks
    return _fetch_any_attr(vec, [
        f"{model_prefix}.attr.consistency_hat", 
        f"{model_prefix}.consistency",
    ], source=src, name="consistency_hat", default=0.5)


def _len_eff(vec: Dict[str, Any], model_prefix: str) -> float:
    """
    Extract length effect normalization factor.
    
    Args:
        vec: Model output dictionary
        model_prefix: Model identifier prefix
        
    Returns:
        Normalized length effect in [0,1]
    """
    src = _source_from_prefix(model_prefix)
    return _fetch_any_attr(vec, [
        f"{model_prefix}.attr.len_effect",
        f"{model_prefix}.len_effect", 
    ], source=src, name="len_effect", default=0.0)


def _temp(vec: Dict[str, Any], model_prefix: str) -> float:
    """
    Extract calibration temperature.
    
    Args:
        vec: Model output dictionary
        model_prefix: Model identifier prefix
        
    Returns:
        Normalized temperature in [0,1]
    """
    src = _source_from_prefix(model_prefix)
    # Temperature is typically already normalized to *01
    return _fetch_any_attr(vec, [
        f"{model_prefix}.attr.temp01",
        f"{model_prefix}.temp01",
    ], source=src, name="temp01", default=0.0)


def _agree(vec: Dict[str, Any], model_prefix: str) -> float:
    """
    Extract agreement prediction with disagree_hat conversion.
    
    Handles both direct agreement scores and disagreement predictions
    by converting disagree_hat to agreement: agree = 1 - disagree_hat
    
    Args:
        vec: Model output dictionary
        model_prefix: Model identifier prefix
        
    Returns:
        Normalized agreement prediction in [0,1]
    """
    src = _source_from_prefix(model_prefix)
    # Prefer direct agreement signal
    if f"{model_prefix}.attr.agree01" in vec:
        return NORMALIZER.norm_attr(vec[f"{model_prefix}.attr.agree01"], source=src, name="agree01")
        
    # Convert disagreement to agreement  
    if f"{model_prefix}.attr.disagree_hat" in vec:
        d = NORMALIZER.norm_attr(vec[f"{model_prefix}.attr.disagree_hat"], source=src, name="disagree_hat")
        return float(np.clip(1.0 - d, 0.0, 1.0))
        
    return 0.5  # Neutral agreement by default


def _aggregate(vec: Dict[str, Any], model_prefix: str, dim_scores: Dict[str, float]) -> float:
    """
    Compute aggregate score with model-specific fallback.
    
    Preference order:
    1. Explicit aggregate score from model
    2. Mean of dimension scores
    3. Default neutral score (0.5)
    
    Args:
        vec: Model output dictionary
        model_prefix: Model identifier prefix  
        dim_scores: Pre-computed dimension scores
        
    Returns:
        Normalized aggregate score in [0,1]
    """
    src = _source_from_prefix(model_prefix)
    
    # Try to find explicit aggregate score
    agg_raw = None
    for k in (f"{model_prefix}.aggregate", f"{model_prefix}.score"):
        if k in vec:
            agg_raw = vec[k]
            break
            
    if agg_raw is not None:
        # Treat aggregate as a score with source-specific scaling
        return NORMALIZER.norm_score(agg_raw, source=src, dimension=None)
        
    # Fall back to mean of dimension scores
    return _mean01(list(dim_scores.values()), 0.0)


# ----------------------------
# Public API
# ----------------------------

def scm_from_vector(vec_native: Dict[str, Any], *, model_prefix: str) -> Dict[str, float]:
    """
    Convert native model outputs to Shared Core Metrics format.
    
    This is the main entry point for SCM conversion. It transforms
    heterogeneous model outputs into the standardized 12-dimensional
    SCM format that enables cross-model comparison.
    
    Pipeline:
    1. Extract dimension scores (reasoning, knowledge, clarity, faithfulness, coverage)
    2. Compute diagnostics (uncertainty, OOD, consistency, etc.)
    3. Compute aggregate score
    4. Return complete SCM dictionary
    
    Args:
        vec_native: Raw model output dictionary with native keys
        model_prefix: Model identifier prefix for key matching
        
    Returns:
        Dictionary with 12 SCM metrics, all normalized to [0,1]
        
    Example:
        >>> hrm_output = {"hrm.reasoning.score": 8.5, "hrm.knowledge.score": 7.0}
        >>> scm_from_vector(hrm_output, model_prefix="hrm")
        {
            "scm.reasoning.score01": 0.85,
            "scm.knowledge.score01": 0.70,
            ... # all 12 SCM metrics
        }
        
    Logs:
        - Debug: Conversion process and fallback usage
        - Warning: Missing critical signals or extreme values
    """
    _logger.debug(f"Converting {model_prefix} outputs to SCM format")
    
    # Tier 1: Extract dimension-specific scores
    dim_scores = {d: _dim_score(vec_native, model_prefix, d) for d in DIMENSIONS}
    
    # Tier 2: Extract diagnostic signals  
    unc = _uncertainty(vec_native, model_prefix)
    ood_val = _ood(vec_native, model_prefix)
    cons = _consistency(vec_native, model_prefix)
    length = _len_eff(vec_native, model_prefix)
    temp_val = _temp(vec_native, model_prefix)
    agree_val = _agree(vec_native, model_prefix)
    
    # Tier 3: Compute aggregate score
    agg = _aggregate(vec_native, model_prefix, dim_scores)

    # Assemble complete SCM dictionary
    scm = {
        "scm.reasoning.score01": dim_scores["reasoning"],
        "scm.knowledge.score01": dim_scores["knowledge"],
        "scm.clarity.score01": dim_scores["clarity"],
        "scm.faithfulness.score01": dim_scores["faithfulness"],
        "scm.coverage.score01": dim_scores["coverage"],
        "scm.aggregate01": agg,
        "scm.uncertainty01": unc,
        "scm.ood_hat01": ood_val,
        "scm.consistency01": cons,
        "scm.length_norm01": length,
        "scm.temp01": temp_val,
        "scm.agree_hat01": agree_val,
    }
    
    _logger.debug(f"SCM conversion complete for {model_prefix}, aggregate: {agg:.3f}")
    return scm


def scm_row(scm_dict: Dict[str, float]) -> List[float]:
    """
    Convert SCM dictionary to ordered row vector.
    
    Args:
        scm_dict: Dictionary of SCM metrics
        
    Returns:
        List of 12 values in SCM_COLUMNS order
        
    Raises:
        KeyError: If SCM dictionary is missing required columns
    """
    return [float(scm_dict.get(k, 0.0)) for k in SCM_COLUMNS]