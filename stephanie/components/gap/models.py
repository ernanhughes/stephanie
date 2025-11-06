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
from typing import Any, Dict, List, Literal, Optional


@dataclass
class EgBadgeConfig:
    """Controls the live risk/hallucination badge in SIS."""
    enabled: bool = True
    # How the compact badge is composed
    show_risk_band: bool = True       # LOW/MED/HIGH chip
    show_hrm_verified: bool = True    # 🛡️ mark when HRM+RAG verified
    show_topo_signal: bool = True     # β₁ / topo_hallucination hint
    show_history_span: int = 50       # turns to summarize in conversation badge
    # Colors (string tokens that your front-end maps to theme vars)
    color_low: str = "green"
    color_med: str = "orange"
    color_high: str = "red"

@dataclass
class EgRenderConfig:
    """Rendering controls for VPMs/strips/GIFs."""
    out_dir: Path = Path("./static/eg_runs")
    make_vpm: bool = True
    make_strip: bool = True
    make_field: bool = True
    make_gif: bool = True
    gif_stride: int = 1
    gif_duration_s: float = 0.8
    # overlays
    show_holes: bool = True
    min_persistence: float = 0.30  # for H1 cycles
    # rasterization (grid for the tensor-first path)
    raster_size: int = 64

@dataclass
class EgThresholds:
    """Routing & alert thresholds."""
    # global fallbacks; domain-calibrated thresholds override at runtime
    risk_low: float = 0.20
    risk_high: float = 0.60
    topo_hallucination_gate: float = 0.55  # if >=, force HRM+RAG
    max_energy_alert: float = 0.80
    wasserstein_h1_warn: float = 0.50

@dataclass
class EgStreams:
    """NATS / telemetry subjects and durable names."""
    nats_url: str = "nats://nats:4222"
    sub_topics: List[str] = field(default_factory=lambda: [
        "qa.scored", "trace.completed"
    ])
    pub_vpm_prefix: str = "vpm.hallucination"
    pub_alerts_topic: str = "alerts.hallucination"
    durable: str = "eg_gap_worker"

@dataclass
class EgMemConfig:
    """MemCube integration."""
    push_evidence: bool = True
    evidence_type: str = "hallucination_map"
    # where to store calibration / domain thresholds, etc.
    store_calibration: bool = True

@dataclass
class EgModelConfig:
    """Models plugged into EG: HalVis, risk, topology."""
    # HalVis backend names you already wired
    halvis_model: str = "halvis_v1"
    # Risk predictor
    risk_model: str = "xgb_isotonic"
    use_domain_calibration: bool = True
    # Topological learner checkpoint (optional, for topo features)
    topo_checkpoint: Optional[Path] = None  # e.g. Path("./models/topo/best.ckpt")

@dataclass
class EgBaselineConfig:
    """Topological baseline corpus for Wasserstein/β deltas."""
    baseline_dir: Path = Path("./static/eg_baseline")
    default_name: str = "qa_good"   # maps to qa_good_stack.npz

@dataclass
class EgConfig:
    """Top-level Epistemic Guard block in GAP."""
    enabled: bool = True
    badge: EgBadgeConfig = field(default_factory=EgBadgeConfig)
    render: EgRenderConfig = field(default_factory=EgRenderConfig)
    thresholds: EgThresholds = field(default_factory=EgThresholds)
    streams: EgStreams = field(default_factory=EgStreams)
    mem: EgMemConfig = field(default_factory=EgMemConfig)
    models: EgModelConfig = field(default_factory=EgModelConfig)
    baseline: EgBaselineConfig = field(default_factory=EgBaselineConfig)

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

    enable_epistemic_guard: bool = False  # Whether to run Epistemic Guard
    eg: EgConfig = field(default_factory=EgConfig)
    risk: Dict[str, Any] = field(default_factory=lambda: {
        "bundle_path": "/models/risk/bundle.joblib",  # Path to pre-bundled artifacts (if any)
        "default_domains": ["science", "history", "geography", "tech", "general"],
        "calib_ttl_s": 3600,
        "fallback_low": 0.20,
        "fallback_high": 0.60,
        "domain_seed_config_path": "config/domain/seeds.yaml"
    })

    domain_seed_config_path: str = "config/domain/seeds.yaml"

    comparison_mode: Literal["models", "runs"] = "models"   # NEW

    # When comparison_mode == "runs", these are used:
    left_label: str = "baseline"         # becomes “hrm_label” internally
    right_label: str = "targeted"        # becomes “tiny_label” internally
    left_run_dir: Optional[Path] = None  # e.g. runs/nexus_vpm/8156-baseline
    right_run_dir: Optional[Path] = None # e.g. runs/nexus_vpm/8157-targeted

    # Which metrics to pull/compare from runs (keep flexible; auto-discover if empty)
    ab_metrics: List[str] = field(default_factory=lambda: [
        # put your most stable VPM/vision/text metrics here; will auto-include what’s found
        "utility", "clarity", "coherence",
        "vision_separability", "vision_bridge_risk", "vision_symmetry",
    ])

@dataclass
class TripleSample:
    """
    Individual data point for model comparison analysis.
    
    Represents a single (goal, response) pair across one reasoning dimension
    that will be scored by both HRM and Tiny models for comparison.
    
    Attributes:
        node_id: Unique identifier for this sample (format: "dimension_fingerprint")
        dimension: Which reasoning dimension this sample belongs to
        goal_text: The input prompt or goal text
        output_text: The model response to evaluate
        target_value: Optional ground truth score for training/validation
        fingerprint: Content-based hash for deduplication and tracking
    """

    node_id: str  # Format: "dimension_fingerprint" e.g., "reasoning_abc123"
    dimension: str  # One of: reasoning, knowledge, clarity, faithfulness, coverage
    goal_text: str  # Original user prompt or goal
    output_text: str  # Model-generated response to evaluate
    target_value: Optional[float] = None  # Ground truth score if available
    fingerprint: Optional[str] = None  # Content hash for deduplication

This is your jumper too big for me is it OK yeah always thank you
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