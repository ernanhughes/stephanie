# stephanie/scoring/tiny_scorer.py
"""
Tiny Recursion Model Scorer - Lightweight evaluator with rich diagnostics.

This module implements the scoring interface for Tiny Recursion Models (TRM),
providing fast, recursive quality assessment with comprehensive diagnostic
telemetry. The scorer transforms TRM's internal signals into standardized
Shared Core Metrics (SCM) format for cross-model comparison in GAP analysis.

Key Features:
- Per-dimension model loading and management
- Rich diagnostic extraction (uncertainty, OOD, sensitivity, agreement, etc.)
- SCM alignment for cross-model comparability
- Vector generation for topological analysis
- Flexible attribute verbosity levels (minimal/standard/full)

The TinyScorer serves as the lightweight counterpart to HRM in the GAP
analysis pipeline, enabling efficient model comparison and routing decisions.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import torch

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.model.tiny_recursion import TinyRecursionModel
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.utils.file_utils import load_json

_logger = logging.getLogger(__name__)


class TinyScorer(BaseScorer):
    """
    Tiny Recursion Model scorer for efficient quality evaluation with rich diagnostics.
    
    This scorer uses trained TinyRecursionModel instances to evaluate goal-response
    pairs across multiple reasoning dimensions. It extracts not just quality scores
    but comprehensive diagnostic telemetry including uncertainty estimates,
    out-of-distribution detection, sensitivity analysis, and agreement predictions.
    
    The scorer automatically converts TRM's native outputs into the standardized
    Shared Core Metrics (SCM) format, enabling direct comparison with HRM and
    other evaluation systems in the GAP analysis pipeline.
    
    Attributes:
        model_type: Identifier for scorer type ("tiny")
        embedding_type: Type of embeddings used (shared with HRM)
        dimensions: List of reasoning dimensions to evaluate
        attr_level: Verbosity level for attributes ("minimal"/"standard"/"full")
        models: Dictionary of loaded TRM models per dimension
        model_meta: Metadata for each dimension's model
    """
    
    def __init__(self, cfg, memory, container, logger):
        """
        Initialize TinyScorer with configuration and dependencies.
        
        Args:
            cfg: Configuration dictionary with scorer parameters
            memory: Memory interface for embedding and data access
            container: Dependency injection container
            logger: Structured logging interface
            
        Configuration Parameters:
            target_type: Type of scoring target ("conversation_turn")
            model_path: Base path for model files
            model_version: Version identifier for models
            dimensions: List of dimensions to evaluate
            clip_0_100: Whether to clip scores to 0-100 range
            tiny_attr_level: Attribute verbosity level
        """
        super().__init__(cfg, memory, container, logger)
        _logger.info("Initializing TinyScorer")
        
        self.model_type = "tiny"  # identifies scorer type in results

        # Embedding interface (shared with HRM for cross-model alignment)
        self.embedding_type = self.memory.embedding.name
        self.dim = self.memory.embedding.dim
        _logger.debug(f"Using embedding type: {self.embedding_type}, dimension: {self.dim}")

        # Configuration parameters
        self.target_type = cfg.get("target_type", "conversation_turn")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.dimensions: List[str] = cfg.get("dimensions", [])
        
        # Output scaling configuration
        self.clip_0_100 = cfg.get("clip_0_100", True)
        
        # Attribute verbosity: controls diagnostic detail level
        self.attr_level = (cfg.get("tiny_attr_level") or "standard").lower()
        _logger.debug(f"Attribute level set to: {self.attr_level}")

        # Containers for per-dimension models and metadata
        self.models: Dict[str, TinyRecursionModel] = {}
        self.model_meta: Dict[str, Dict[str, Any]] = {}

        # Attempt to load models up-front for all specified dimensions
        _logger.info(f"Loading TRM models for dimensions: {self.dimensions}")
        self._load_models(self.dimensions)
        _logger.info(f"TinyScorer initialized with {len(self.models)} loaded models")

    # -------------------------
    # Model Loading
    # -------------------------
    def _load_models(self, dimensions: List[str]):
        """
        Load trained TinyRecursionModel instances for specified dimensions.
        
        For each dimension, this method:
        1. Resolves model and metadata file paths
        2. Loads model configuration from metadata
        3. Instantiates TRM with correct architecture
        4. Loads trained weights
        5. Registers model in the internal registry
        
        Args:
            dimensions: List of reasoning dimensions to load models for
            
        Logs:
            - Debug: Model loading progress and configuration
            - Warning: Missing model files or metadata
            - Error: Model instantiation or weight loading failures
        """
        _logger.debug(f"Starting model loading for {len(dimensions)} dimensions")
        
        for dim in dimensions:
            _logger.debug(f"Loading model for dimension: {dim}")
            locator = self.get_locator(dim)

            # Resolve model and metadata file paths
            model_path = locator.model_file(suffix="_tiny.pt")
            meta_path = locator.meta_file()
            _logger.debug(f"Model path: {model_path}, Meta path: {meta_path}")

            if not os.path.exists(model_path):
                _logger.warning(f"Model file missing for dimension {dim}: {model_path}")
                self.logger.log(
                    "TinyScorerModelMissing",
                    {"dimension": dim, "path": model_path},
                )
                continue

            # Load model metadata for architecture configuration
            meta: Dict[str, Any] = {}
            if os.path.exists(meta_path):
                try:
                    meta = load_json(meta_path) or {}
                    _logger.debug(f"Loaded metadata for {dim}: {len(meta)} keys")
                except Exception as e:
                    _logger.error(f"Failed to load metadata for {dim}: {e}")
                    self.logger.log(
                        "TinyScorerMetaLoadError", {"dimension": dim, "error": str(e)}
                    )
            else:
                _logger.warning(f"Metadata file missing for {dim}: {meta_path}")

            # Extract model configuration from metadata with safe defaults
            cfg_meta = meta.get("cfg", {}) if isinstance(meta, dict) else {}
            n_layers = int(cfg_meta.get("n_layers", 2))
            n_recursions = int(cfg_meta.get("n_recursions", 6))
            use_attn = bool(cfg_meta.get("use_attention", False))
            dropout = float(cfg_meta.get("dropout", 0.1))
            attn_heads = int(cfg_meta.get("attn_heads", 4))
            step_scale = float(cfg_meta.get("step_scale", 0.1))
            cons_mask_p = float(cfg_meta.get("consistency_mask_p", 0.10))
            len_norm_L = float(cfg_meta.get("len_norm_L", 512.0))
            vocab_size = int(cfg_meta.get("vocab_size", 101))

            # Optional feature flags from metadata
            enable_agree_head = bool(cfg_meta.get("enable_agree_head", True))
            enable_causal_sens_head = bool(cfg_meta.get("enable_causal_sens_head", True))

            _logger.debug(
                f"Model config for {dim}: layers={n_layers}, recursions={n_recursions}, "
                f"attention={use_attn}, dropout={dropout}"
            )

            # Instantiate model with exact same architecture as training
            _logger.debug(f"Instantiating TRM for dimension {dim}")
            model = TinyRecursionModel(
                d_model=self.dim,
                n_layers=n_layers,
                n_recursions=n_recursions,
                vocab_size=vocab_size,
                use_attention=use_attn,
                dropout=dropout,
                attn_heads=attn_heads,
                step_scale=step_scale,
                consistency_mask_p=cons_mask_p,
                len_norm_L=len_norm_L,
                enable_agree_head=enable_agree_head,
                enable_causal_sens_head=enable_causal_sens_head,
            ).to(self.device)

            # Load trained weights with strict=False for backward compatibility
            _logger.debug(f"Loading model weights from: {model_path}")
            try:
                state = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state, strict=False)
                model.eval()  # Set to evaluation mode
                _logger.debug(f"Successfully loaded weights for {dim}")
            except Exception as e:
                _logger.error(f"Failed to load weights for {dim}: {e}")
                continue

            # Register successfully loaded model
            self.models[dim] = model
            self.model_meta[dim] = meta
            
            _logger.info(f"Successfully loaded TRM model for dimension: {dim}")
            self.logger.log(
                "TinyScorerModelLoaded",
                {
                    "dimension": dim, 
                    "model_path": model_path, 
                    "device": str(self.device)
                },
            )

    # -------------------------
    # Scoring Core
    # -------------------------
    def _score_core(self, context: dict, scorable, dimensions: List[str]) -> ScoreBundle:
        """
        Core scoring method that evaluates goal-response pairs using TRM.
        
        This method:
        1. Converts text to embeddings (shared with HRM)
        2. Runs TRM inference for each dimension
        3. Extracts scores and rich diagnostics
        4. Converts to SCM format for cross-model alignment
        5. Generates aligned vectors for topological analysis
        
        Args:
            context: Scoring context containing goal information
            scorable: The response text to evaluate
            dimensions: List of dimensions to score against
            
        Returns:
            ScoreBundle containing results for all specified dimensions
            
        Logs:
            - Debug: Embedding conversion, model inference, SCM conversion
            - Warning: Missing models or scoring errors
            - Info: Scoring completion statistics
        """
        _logger.debug(f"Starting scoring for {len(dimensions)} dimensions")
        
        # Extract goal information from context
        goal = context.get(GOAL, {})
        goal_text = goal.get(GOAL_TEXT, "")
        _logger.debug(f"Scoring goal: {goal_text[:50]}...")
        _logger.debug(f"Scorable text: {scorable.text[:50]}...")

        results: Dict[str, ScoreResult] = {}

        # Step 1: Convert text to embeddings (shared with HRM for alignment)
        _logger.debug("Converting goal and response to embeddings")
        x_np = self.memory.embedding.get_or_create(goal_text)
        y_np = self.memory.embedding.get_or_create(scorable.text)
        
        # Convert to tensors and ensure correct device placement
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = torch.zeros_like(x)  # neutral third stream for recursive processing
        seq_len = torch.zeros(x.size(0), dtype=torch.int32, device=self.device)
        
        _logger.debug(f"Embedding shapes - x: {x.shape}, y: {y.shape}, z: {z.shape}")

        # Step 2: Score for each specified dimension
        for dim in dimensions:
            _logger.debug(f"Scoring dimension: {dim}")
            model = self.models.get(dim)
            
            if model is None:
                _logger.warning(f"No model available for dimension: {dim}")
                self.logger.log("TinyModelMissing", {"dimension": dim})
                continue

            try:
                # Run TRM inference with gradient disabled for efficiency
                _logger.debug(f"Running TRM inference for {dim}")
                with torch.no_grad():
                    _, halt_logits, _, aux = model(
                        x, y, z, seq_len=seq_len, return_aux=True
                    )
                _logger.debug(f"TRM inference completed for {dim}")

                # Step 3: Extract core metrics from model outputs
                _logger.debug("Extracting core metrics from TRM outputs")
                raw01 = float(max(0.0, min(1.0, _tf(aux.get("score")))))
                
                # Calculate certainty with fallback logic
                if "certainty01" in aux:
                    certainty01 = _tf(aux["certainty01"])
                    _logger.debug("Using certainty01 from aux")
                elif "uncertainty01" in aux:
                    certainty01 = 1.0 - _tf(aux["uncertainty01"])
                    _logger.debug("Derived certainty from uncertainty01")
                elif "uncertainty" in aux:
                    certainty01 = 1.0 - _tf(aux["uncertainty"])
                    _logger.debug("Derived certainty from uncertainty")
                else:
                    certainty01 = 0.5  # Default neutral certainty
                    _logger.debug("Using default certainty 0.5")
                    
                entropy = _tf(aux.get("entropy_aux"))
                halt_prob = _sigmoid_mean(halt_logits)
                
                _logger.debug(
                    f"Core metrics - score: {raw01:.3f}, certainty: {certainty01:.3f}, "
                    f"entropy: {entropy:.3f}, halt_prob: {halt_prob:.3f}"
                )

                # Apply scaling and metadata adjustments
                meta = self.model_meta.get(dim, {})
                final_score = _tf(aux.get("score"))
                tiny_score01 = raw01
                tiny_score100 = round(_safe_scale_0_100(tiny_score01, meta), 4)
                _logger.debug(f"Scaled scores - 01: {tiny_score01:.3f}, 100: {tiny_score100}")

                # Step 4: Build base attributes dictionary
                _logger.debug("Building base attributes dictionary")
                attrs: Dict[str, Any] = {
                    "tiny.score01": tiny_score01,
                    "tiny.score100": tiny_score100,
                    "raw01": tiny_score01,  # backward-compatibility alias
                    "entropy": float(entropy),
                    "certainty01": float(certainty01),
                    "halt_prob": float(halt_prob) if halt_prob is not None else None,
                    # Model context metadata for downstream processing
                    "n_recursions": int(meta.get("cfg", {}).get("n_recursions", 6)),
                    "use_attention": bool(meta.get("cfg", {}).get("use_attention", False)),
                    "dropout": float(meta.get("cfg", {}).get("dropout", 0.1)),
                }

                # Step 5: Add diagnostic attributes based on verbosity level
                if self.attr_level in ("standard", "full"):
                    _logger.debug("Adding standard diagnostic attributes")
                    attrs.update(_extract_standard_aux(aux))
                    
                    # Include optional bridge heads if available
                    if "agree01" in aux and isinstance(aux["agree01"], torch.Tensor):
                        attrs["agree01"] = float(_tf(aux["agree01"]))
                        _logger.debug("Added agree01 diagnostic")
                    if "sens01" in aux and isinstance(aux["sens01"], torch.Tensor):
                        attrs["sens01"] = float(_tf(aux["sens01"]))
                        _logger.debug("Added sens01 diagnostic")

                if self.attr_level == "full":
                    _logger.debug("Adding full diagnostic attributes")
                    attrs.update(_extract_full_aux(aux))
                    
                    # Add raw signal summaries for deep debugging
                    if "score_logit" in aux:
                        attrs["score_logit_mean"] = float(_tf(aux["score_logit"]))
                    if "aux3_logits" in aux and isinstance(aux["aux3_logits"], torch.Tensor):
                        al = aux["aux3_logits"]
                        attrs["aux3_logits_l1_mean"] = float(al.abs().mean().item())

                # Step 6: Convert to Shared Core Metrics format
                _logger.debug("Converting to SCM format for cross-model alignment")
                scm = _build_scm_from_tiny_attrs(attrs)
                attrs.update(scm)
                _logger.debug(f"SCM conversion complete - aggregate: {scm.get('scm.aggregate01', 0):.3f}")

                # Step 7: Mirror dimension scores for PHOS compatibility
                _logger.debug("Mirroring dimension scores for PHOS alignment")
                for dname in ("reasoning", "knowledge", "clarity", "faithfulness", "coverage"):
                    key = f"scm.{dname}.score01"
                    if key in scm:
                        v01 = float(scm[key])
                        attrs[f"tiny.{dname}.score01"]  = v01
                        attrs[f"tiny.{dname}.score100"] = round(v01 * 100.0, 4)
                        attrs[f"tiny.{dname}"] = float(scm[key])
                _logger.debug("Dimension score mirroring complete")

                # Step 8: Generate scoring rationale
                rationale = (
                    f"tiny[{dim}] raw01={float(raw01):.4f}, "
                    f"H={float(entropy):.3f}, C={float(certainty01):.3f}, "
                    f"halt_p={float(halt_prob) if halt_prob is not None else -1:.3f}"
                )
                _logger.debug(f"Generated rationale: {rationale}")

                # Step 9: Build aligned vector for topological analysis
                _logger.debug("Building aligned vector for GAP analysis")
                vector = _tiny_build_vector(attrs)

                # Step 10: Create final ScoreResult
                results[dim] = ScoreResult(
                    dimension=dim,
                    score=tiny_score01,
                    source=self.model_type,
                    rationale=rationale,
                    weight=1.0,
                    attributes={
                        **attrs, 
                        "vector": vector["vector"], 
                        "columns": vector["columns"], 
                        "values": vector["values"]
                    },
                )
                _logger.debug(f"Successfully created ScoreResult for {dim}")

            except Exception as e:
                _logger.error(f"Scoring error for dimension {dim}: {e}")
                self.logger.log("TinyScoreError", {"dimension": dim, "error": str(e)})

        _logger.info(f"Scoring completed for {len(results)} dimensions")
        return ScoreBundle(results=results)

    # -------------------------
    # Utility Methods
    # -------------------------
    @staticmethod
    def _get(d: Dict[str, Any], key: str):
        """
        Safe dictionary access with exception handling.
        
        Args:
            d: Dictionary to access
            key: Key to retrieve
            
        Returns:
            Value if present and accessible, None otherwise
        """
        try:
            return d.get(key)
        except Exception:
            return None

    def __repr__(self):
        """String representation showing loaded models."""
        loaded = {k: (v is not None) for k, v in self.models.items()}
        return f"<TinyScorer(model_type={self.model_type}, loaded={loaded})>"


def _take_scalar(t):
    """
    Extract scalar value from tensor or return float directly.
    
    Args:
        t: Input tensor or scalar
        
    Returns:
        Extracted scalar value as float
    """
    # works with tensor or float
    if isinstance(t, torch.Tensor):
        return float(t.detach().mean().cpu().item())
    return float(t)


# -------------------------
# Helper Functions
# -------------------------

def _tf(v):
    """
    Tensor/array/number → scalar float with safe fallback.
    
    Handles various input types and extracts mean value from tensors.
    Provides safe defaults for None or invalid inputs.
    
    Args:
        v: Input value (tensor, array, or scalar)
        
    Returns:
        Extracted scalar float value
    """
    if v is None:
        _logger.debug("Received None value, returning 0.0")
        return 0.0
    if isinstance(v, torch.Tensor):
        # handle both scalar and vector tensors - use mean for vectors
        result = v.detach().float().mean().item()
        _logger.debug(f"Converted tensor to scalar: {result}")
        return result
    try:
        result = float(v)
        _logger.debug(f"Converted value to float: {result}")
        return result
    except Exception:
        _logger.debug(f"Failed to convert value: {v}, returning 0.0")
        return 0.0


def _sigmoid_mean(v):
    """
    Apply sigmoid and compute mean for halting logits.
    
    Args:
        v: Input tensor or value
        
    Returns:
        Mean sigmoid probability, or None if input is None
    """
    if v is None:
        _logger.debug("Received None for sigmoid_mean")
        return None
    if isinstance(v, torch.Tensor):
        result = torch.sigmoid(v.detach()).mean().item()
        _logger.debug(f"Computed sigmoid mean: {result}")
        return result
    result = float(v)
    _logger.debug(f"Returning float value: {result}")
    return result


def _safe_scale_0_100(raw: float, meta: dict | None) -> float:
    """
    Scale raw [0,1] score to [0,100] range with metadata awareness.
    
    Uses metadata min/max values if available, otherwise uses default 0-100 scaling.
    
    Args:
        raw: Raw score in [0,1] range
        meta: Model metadata containing scaling parameters
        
    Returns:
        Scaled score in appropriate range
    """
    if not meta:
        result = float(max(0.0, min(1.0, raw)) * 100.0)
        _logger.debug(f"Scaled without meta: {raw} -> {result}")
        return result
    
    lo = float(meta.get("min_value", 0.0))
    hi = float(meta.get("max_value", 100.0))
    result = float(max(lo, min(hi, lo + (hi - lo) * max(0.0, min(1.0, raw)))))
    _logger.debug(f"Scaled with meta: {raw} -> {result} (range: {lo}-{hi})")
    return result


def _tiny_build_vector(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build aligned vector representation for GAP analysis.
    
    Creates a deterministic vector structure that enables cross-model
    alignment in topological analysis. Includes both raw TRM statistics
    and SCM-formatted metrics.
    
    Args:
        attrs: Dictionary of attributes from TRM scoring
        
    Returns:
        Dictionary containing vector, columns, and values for alignment
    """
    _logger.debug("Building aligned vector from attributes")
    vec: Dict[str, float] = {}
    
    # Core TRM statistics for direct access
    vec["tiny.score01"]        = float(attrs.get("tiny.score01", 0.0))
    vec["tiny.score100"]       = float(attrs.get("tiny.score100", 0.0))
    vec["tiny.certainty01"]    = float(attrs.get("certainty01", 0.5))
    vec["tiny.entropy_mean"]   = float(attrs.get("entropy", 0.0))
    
    if "halt_prob" in attrs and attrs["halt_prob"] is not None:
        vec["tiny.halt_prob"] = float(attrs["halt_prob"])
    
    _logger.debug(f"Added {len(vec)} core TRM statistics to vector")

    # SCM-formatted metrics for cross-model alignment
    scm_keys = [
        "scm.reasoning.score01", "scm.knowledge.score01", "scm.clarity.score01",
        "scm.faithfulness.score01", "scm.coverage.score01", "scm.aggregate01",
        "scm.uncertainty01", "scm.ood_hat01", "scm.consistency01",
        "scm.length_norm01", "scm.temp01", "scm.agree_hat01",
    ]
    
    scm_count = 0
    for k in scm_keys:
        if k in attrs:
            vec[k] = float(attrs[k])
            scm_count += 1
    
    _logger.debug(f"Added {scm_count} SCM metrics to vector")

    # Mirror dimension scores for PHOS visualization compatibility
    mirror_count = 0
    for d in ("reasoning", "knowledge", "clarity", "faithfulness", "coverage"):
        k = f"scm.{d}.score01"
        if k in attrs:
            v01 = float(attrs[k])
            vec[f"tiny.{d}.score01"]  = v01
            vec[f"tiny.{d}.score100"] = round(v01 * 100.0, 4)
            vec[f"tiny.{d}"]          = v01
            mirror_count += 1
    
    _logger.debug(f"Mirrored {mirror_count} dimension scores")

    # Create final aligned structure
    cols = list(vec.keys())
    vals = [vec[c] for c in cols]
    _logger.debug(f"Vector construction complete: {len(cols)} columns, {len(vals)} values")
    
    return {"vector": vec, "columns": cols, "values": vals}


def _extract_standard_aux(aux: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract standard diagnostic attributes from TRM auxiliary outputs.
    
    Provides balanced diagnostic coverage including confidence estimates,
    calibration signals, robustness measures, and sensitivity analysis.
    All outputs are normalized to [0,1] range.
    
    Args:
        aux: TRM auxiliary outputs dictionary
        
    Returns:
        Dictionary of standardized diagnostic attributes
    """
    _logger.debug("Extracting standard diagnostic attributes")
    out: Dict[str, float] = {}

    # Confidence triplet from 3-class auxiliary head
    if "aux3_probs" in aux and isinstance(aux["aux3_probs"], torch.Tensor):
        p = aux["aux3_probs"].detach().float()
        out["aux3_p_bad"]  = float(p[..., 0].mean().item())
        out["aux3_p_mid"]  = float(p[..., 1].mean().item())
        out["aux3_p_good"] = float(p[..., 2].mean().item())
        _logger.debug("Extracted aux3 probability triplet")

    # Calibration and temperature signals
    out["temp01"] = float(_tf(aux.get("temp01")))

    # Out-of-distribution detection (prefer newer ood_hat01 format)
    if "ood_hat01" in aux:
        out["ood_hat01"] = float(_tf(aux["ood_hat01"]))
        _logger.debug("Using ood_hat01 for OOD detection")
    elif "ood_hat" in aux:  # backward compatibility
        out["ood_hat01"] = float(_tf(aux["ood_hat"]))
        _logger.debug("Using ood_hat (legacy) for OOD detection")

    # Robustness and sensitivity measures
    out["consistency_hat"] = float(_tf(aux.get("consistency_hat")))
    out["jacobian_fd"]     = float(_tf(aux.get("jacobian_fd")))
    _logger.debug("Extracted robustness and sensitivity measures")

    # Reconstruction quality and disagreement prediction
    out["recon_sim"]  = float(_tf(aux.get("recon_sim")))
    out["disagree_hat"] = float(_tf(aux.get("disagree_hat")))
    _logger.debug("Extracted reconstruction and disagreement signals")

    # Length normalization (prefer 0..1 normalized version)
    if "length_norm01" in aux:
        out["length_norm01"] = float(_tf(aux["length_norm01"]))
        _logger.debug("Using length_norm01 for length effect")
    else:
        # Derive from tanh-normalized len_effect if available
        if "len_effect" in aux:
            le = float(_tf(aux["len_effect"]))
            out["length_norm01"] = float(max(0.0, min(1.0, (le + 1.0) * 0.5)))
            _logger.debug("Derived length_norm01 from len_effect")
        else:
            out["length_norm01"] = 0.0
            _logger.debug("Using default length_norm01")

    # Sparse Autoencoder concept sparsity
    out["concept_sparsity"] = float(_tf(aux.get("concept_sparsity")))
    _logger.debug("Extracted concept sparsity measure")

    _logger.debug(f"Standard diagnostics extraction complete: {len(out)} attributes")
    return out


def _extract_full_aux(aux: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract full diagnostic attributes including raw signal summaries.
    
    Provides maximum detail for debugging and analysis, including
    raw logit summaries and internal representation statistics.
    Use only when deep inspection is required.
    
    Args:
        aux: TRM auxiliary outputs dictionary
        
    Returns:
        Dictionary of detailed diagnostic attributes
    """
    _logger.debug("Extracting full diagnostic attributes")
    out: Dict[str, float] = {}

    # Summaries of raw head outputs for debugging
    for k in ("log_var", "consistency_logit", "disagree_logit"):
        if k in aux and isinstance(aux[k], torch.Tensor):
            t = aux[k].detach()
            out[f"{k}_mean"] = float(t.mean().item())
            _logger.debug(f"Added {k}_mean to full diagnostics")

    # Reconstruction detail analysis
    if "y_recon" in aux and isinstance(aux["y_recon"], torch.Tensor):
        yr = aux["y_recon"].detach()
        out["y_recon_norm_mean"] = float(yr.norm(dim=-1).mean().item())
        _logger.debug("Added y_recon_norm_mean to full diagnostics")

    # Sparse Autoencoder concept analysis
    if "concept_vec" in aux and isinstance(aux["concept_vec"], torch.Tensor):
        c = aux["concept_vec"].detach()
        out["concept_vec_l2_mean"] = float((c.pow(2).sum(-1).sqrt()).mean().item())
        _logger.debug("Added concept_vec_l2_mean to full diagnostics")

    _logger.debug(f"Full diagnostics extraction complete: {len(out)} attributes")
    return out


# === SCM mapping from Tiny aux → aligned scm.* columns =======================

def _build_scm_from_tiny_attrs(attrs: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert TRM attributes to Shared Core Metrics format.
    
    Maps TRM's internal diagnostic signals to the standardized 5-dimensional
    SCM format using learned weighting patterns. This enables direct
    comparison with HRM and other evaluation systems.
    
    The mapping uses TRM's diagnostic patterns to infer dimension scores:
    - Reasoning: Emphasizes consistency, low uncertainty, agreement
    - Knowledge: Focuses on in-distribution signals and reconstruction
    - Clarity: Uses token quality and length normalization
    - Faithfulness: Based on reconstruction and consistency
    - Coverage: Considers concept activity and distribution alignment
    
    Args:
        attrs: TRM attributes dictionary
        
    Returns:
        Dictionary of SCM-formatted scores in [0,1] range
    """
    _logger.debug("Building SCM from TRM attributes")
    
    # Extract and clamp core diagnostic signals
    certainty = float(attrs.get("certainty01", 0.5))
    unc01     = 1.0 - max(0.0, min(1.0, certainty))
    cons01    = max(0.0, min(1.0, float(attrs.get("consistency_hat", 0.5))))
    ood01     = max(0.0, min(1.0, float(attrs.get("ood_hat", 0.0))))
    len01     = max(0.0, min(1.0, float(attrs.get("len_effect", 0.0))))
    temp01    = max(0.0, min(1.0, float(attrs.get("temp01", 0.0))))
    agree01   = max(0.0, min(1.0, float(attrs.get("agree01", 0.5))))

    # Extract additional diagnostic signals
    recon_sim      = max(0.0, min(1.0, float(attrs.get("recon_sim", 0.5))))
    concept_sparse = max(0.0, min(1.0, float(attrs.get("concept_sparsity", 0.5))))
    p_bad          = max(0.0, min(1.0, float(attrs.get("aux3_p_bad", 0.5))))
    token_ok       = 1.0 - p_bad  # clarity proxy: lower bad probability → clearer

    _logger.debug(f"Core signals - uncertainty: {unc01:.3f}, consistency: {cons01:.3f}, OOD: {ood01:.3f}")

    # Dimension-specific scoring using diagnostic patterns
    dim_scores: Dict[str, float] = {}
    
    # Reasoning: weighted toward stability, consistency, and confidence
    dim_scores["reasoning"] = 0.60*cons01 + 0.30*(1.0-unc01) + 0.10*agree01
    _logger.debug(f"Reasoning score: {dim_scores['reasoning']:.3f}")
    
    # Knowledge: emphasizes distribution alignment and comprehension
    dim_scores["knowledge"] = 0.50*(1.0-ood01) + 0.30*recon_sim + 0.20*(1.0-unc01)
    _logger.debug(f"Knowledge score: {dim_scores['knowledge']:.3f}")
    
    # Clarity: based on token quality and brevity
    dim_scores["clarity"] = 0.50*token_ok + 0.30*(1.0-len01) + 0.20*cons01
    _logger.debug(f"Clarity score: {dim_scores['clarity']:.3f}")
    
    # Faithfulness: reconstruction quality and stability
    dim_scores["faithfulness"] = 0.50*recon_sim + 0.30*cons01 + 0.20*(1.0-unc01)
    _logger.debug(f"Faithfulness score: {dim_scores['faithfulness']:.3f}")
    
    # Coverage: concept activity and confidence
    dim_scores["coverage"] = 0.40*concept_sparse + 0.40*(1.0-unc01) + 0.20*(1.0-ood01)
    _logger.debug(f"Coverage score: {dim_scores['coverage']:.3f}")

    # Ensure all scores are in valid [0,1] range
    for k in dim_scores:
        v = dim_scores[k]
        dim_scores[k] = float(min(1.0, max(0.0, v)))
    _logger.debug("Applied score clamping to [0,1] range")

    # Build final SCM dictionary
    scm: Dict[str, float] = {
        f"scm.{k}.score01": dim_scores[k]
        for k in ("reasoning", "knowledge", "clarity", "faithfulness", "coverage")
    }
    scm["scm.aggregate01"]   = float(sum(dim_scores.values())/5.0)
    scm["scm.uncertainty01"] = float(unc01)
    scm["scm.ood_hat01"]     = float(ood01)
    scm["scm.consistency01"] = float(cons01)
    scm["scm.length_norm01"] = float(len01)
    scm["scm.temp01"]        = float(temp01)
    scm["scm.agree_hat01"]   = float(agree01)

    _logger.debug(f"SCM construction complete - aggregate: {scm['scm.aggregate01']:.3f}")
    return scm
