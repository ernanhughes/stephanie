# stephanie/components/gap/models.py
"""
Data models for GAP (Gap Analysis Project) component.

This module defines the core data structures used throughout the GAP analysis pipeline:
- GapConfig: Configuration and parameters for analysis runs
- TripleSample: Individual data points for model comparison  
- ModelScores: Results from model evaluations across dimensions

These models provide type safety and structure for the complex data flow between
scoring, analysis, and visualization components in the GAP system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GapConfig:
    """
    Configuration container for GAP analysis runs.
    
    Defines all tunable parameters, paths, and behavior switches for comparing
    HRM vs Tiny models across reasoning dimensions.
    
    Attributes:
        dimensions: Reasoning facets to evaluate (5 core dimensions)
        hrm_scorers: List of scorer names for HRM model evaluation
        tiny_scorers: List of scorer names for Tiny model evaluation  
        out_dir: Primary output directory for analysis artifacts
        base_dir: Root directory for GAP run storage
        interleave: Whether to mix samples across dimensions during processing
        progress_log_every: Frequency of progress logging (in samples)
        dedupe_policy: Strategy for handling duplicate samples ("first_wins")
        per_dim_cap: Maximum samples per dimension to prevent memory issues
        route_threshold_uncertainty: Uncertainty level for model routing decisions
        route_threshold_ood: Out-of-distribution threshold for routing
        enable_scm_head: Whether to compute Shared Core Metrics projections
        scm: Configuration for SCM (Shared Core Metrics) processing
    """
    
    # Core analysis dimensions - the five reasoning facets
    dimensions: List[str] = field(default_factory=lambda: [
        "reasoning", "knowledge", "clarity", "faithfulness", "coverage"
    ])
    
    # Model scorer configurations - support both HF and local implementations
    hrm_scorers: List[str] = field(default_factory=lambda: ["hf_hrm"])
    tiny_scorers: List[str] = field(default_factory=lambda: ["hf_mistral"])
    # Alternative local scorer configurations:
    # hrm_scorers: List[str] = field(default_factory=lambda: ["hrm"])
    # tiny_scorers: List[str] = field(default_factory=lambda: ["tiny"])
    
    # Directory structure for artifact storage
    out_dir: Path = field(default_factory=lambda: Path("data/gap_runs/vpm"))
    base_dir: Path = field(default_factory=lambda: Path("data/gap_runs"))
    
    # Processing behavior controls
    interleave: bool = False  # Mix samples across dimensions
    progress_log_every: int = 25  # Log every N samples for progress tracking
    dedupe_policy: str = "first_wins"  # Duplicate handling: "first_wins", "average", "reject"
    
    # Resource management - control dataset size for memory/performance
    per_dim_cap: int = 1000  # Maximum samples per dimension for production
    # per_dim_cap: int = 100  # Reduced cap for development/testing
    
    # Routing thresholds for model escalation decisions
    route_threshold_uncertainty: float = 0.6  # Escalate if uncertainty > 60%
    route_threshold_ood: float = 0.7  # Escalate if OOD detection > 70%
    
    # Advanced features
    enable_scm_head: bool = True  # Enable Shared Core Metrics computation
    
    # SCM (Shared Core Metrics) configuration
    scm: Dict[str, Any] = field(default_factory=lambda: {
        "reasoning_dimensions": ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"],
        "latent_dim": 128,  # Dimension of SCM latent space projections
    })


@dataclass
class TripleSample:
    """
    Individual data point for model comparison analysis.
    
    Represents a single (goal, response) pair across one reasoning dimension
    that will be scored by both HRM and Tiny models for comparison.
    
    Attributes:
        node_id: Unique identifier for this sample (format: "dimension|fingerprint")
        dimension: Which reasoning dimension this sample belongs to
        goal_text: The input prompt or goal text
        output_text: The model response to evaluate
        target_value: Optional ground truth score for training/validation
        fingerprint: Content-based hash for deduplication and tracking
    """
    
    node_id: str  # Format: "dimension|fingerprint" e.g., "reasoning|abc123"
    dimension: str  # One of: reasoning, knowledge, clarity, faithfulness, coverage
    goal_text: str  # Original user prompt or goal
    output_text: str  # Model-generated response to evaluate
    target_value: Optional[float] = None  # Ground truth score if available
    fingerprint: Optional[str] = None  # Content hash for deduplication


@dataclass  
class ModelScores:
    """
    Container for model evaluation results across multiple dimensions.
    
    Stores scores, metrics, and vector representations from a single model
    (HRM or Tiny) for a set of samples. Used for comparison and delta analysis.
    
    Attributes:
        model_name: Identifier for the model (e.g., "hrm", "tiny", "hf_mistral")
        scores: Dictionary mapping dimension names to quality scores
        metrics: Additional diagnostic metrics (uncertainty, OOD, consistency, etc.)
        vector: Optional flattened feature vector for alignment and visualization
    """
    
    model_name: str  # Model identifier: "hrm", "tiny", or HF model name
    scores: Dict[str, float]  # Dimension -> score mapping
    metrics: Dict[str, Any] = field(default_factory=dict)  # Diagnostic telemetry
    vector: Optional[List[float]] = None  # Aligned feature vector for analysis