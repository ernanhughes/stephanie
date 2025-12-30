# stephanie/scoring/scorer/mrq_scorer.py Can I go to bed I
"""
MRQ (Multi-dimensional Response Quality) Scorer Module

This module implements scoring for multi-dimensional response quality assessment.
It uses trained encoder-predictor models to evaluate text pairs (goal, response)
across multiple quality dimensions, then scales the results to a standardized range.

Key components:
- MRQModel: Neural model for quality prediction
- TextEncoder: Encodes text pairs into embeddings
- ValuePredictor: Predicts quality scores from embeddings
- RegressionTuner: Calibrates raw predictions to target scales

The scorer handles model loading, inference, and score normalization across dimensions.
"""

from __future__ import annotations

import os
from typing import Dict, List, Union

import torch

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.model.mrq_model import MRQModel
from stephanie.model.text_encoder import TextEncoder
from stephanie.model.value_predictor import ValuePredictor
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator


class MRQScorer(BaseScorer):
    """
    Multi-dimensional Response Quality Scorer
    
    Computes pairwise quality scores for (goal, response) pairs using encoder-predictor models.
    For each dimension, produces a logit that is scaled to the dimension's target range using
    either a RegressionTuner (calibrated probability mapping) or sigmoid with linear scaling.
    
    Features:
    - Supports multiple quality dimensions (e.g., relevance, coherence, fluency)
    - Handles model loading and versioning
    - Provides score normalization and legacy scale conversion
    - Maintains embedding cache for efficient computation
    """

    def __init__(self, cfg, memory, container, logger):
        """Initialize MRQ scorer with configuration and dependencies"""
        super().__init__(cfg, memory, container, logger)
        self.model_type = "mrq"
        self.embedding_type = self.memory.embedding.name
        self.dim = memory.embedding.dim  # Input dimension for encoder
        self.hdim = memory.embedding.hdim  # Hidden dimension for model

        # Configuration parameters
        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")

        # Model and calibration storage
        self.models: Dict[str, MRQModel] = {}  # Dimension -> model mapping
        self.model_meta: Dict[str, dict] = {}  # Model metadata (min/max values)
        self.tuners: Dict[str, RegressionTuner] = {}  # Calibration tuners

        # Dimensions to score
        self.dimensions: List[str] = cfg.get("dimensions", [])
        self._load_models(self.dimensions)

    # ---------- model loading and management ----------

    def _locator(self, dim: str) -> ModelLocator:
        """Create model locator for specific dimension"""
        return ModelLocator(
            root_dir=self.model_path,
            embedding_type=self.embedding_type,
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dim,
            version=self.version,
        )

    def _load_models(self, dimensions: List[str]) -> None:
        """Load models and calibration tuners for all dimensions"""
        for dim in dimensions:
            loc = self._locator(dim)

            # Build model architecture matching the training setup
            encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
            predictor = ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)

            # Load trained weights
            encoder.load_state_dict(torch.load(loc.encoder_file(), map_location=self.device))
            predictor.load_state_dict(torch.load(loc.model_file(), map_location=self.device))

            # Create complete MRQ model
            model = MRQModel(
                encoder=encoder,
                predictor=predictor,
                embedding_store=self.memory.embedding,
                device=self.device,
            )
            self.models[dim] = model

            # Load model metadata and calibration tuner if available
            meta = load_json(loc.meta_file()) if os.path.exists(loc.meta_file()) else {"min_value": 0.0, "max_value": 100.0}
            self.model_meta[dim] = meta

            tuner_path = loc.tuner_file()
            if os.path.exists(tuner_path):
                t = RegressionTuner(dimension=dim)
                t.load(tuner_path)
                self.tuners[dim] = t

    # ---------- score scaling and normalization ----------

    def _scale(self, dim: str, logit: float, meta: dict) -> float:
        """
        Convert raw model logit to normalized score in [0, 1] range
        
        Process:
        1. Apply sigmoid to get probability (0-1)
        2. Use RegressionTuner if available, else linear scaling with meta min/max
        3. Detect and handle legacy 0-100 scales, converting to 0-1
        4. Clamp final result to valid [0, 1] range
        
        Args:
            dim: Dimension name for tuner lookup
            logit: Raw model output
            meta: Model metadata with min/max values
            
        Returns:
            Normalized score in [0, 1] range
        """
        # Convert logit to probability using sigmoid
        prob = torch.sigmoid(torch.tensor(logit, dtype=torch.float32)).item()

        # Apply scaling: use tuner if available, else linear scaling
        if dim in self.tuners:
            val_raw = float(self.tuners[dim].transform(prob))
            min_v = float(meta.get("min_value", 0.0))
            max_v = float(meta.get("max_value", 1.0))
        else:
            min_v = float(meta.get("min_value", 0.0))
            max_v = float(meta.get("max_value", 1.0))
            val_raw = prob * (max_v - min_v) + min_v

        # Handle legacy 0-100 scale models by detecting values > 1.01
        if max_v > 1.01 or val_raw > 1.01:
            # Convert from 0-100 scale to 0-1
            span = (max_v - min_v) if (max_v - min_v) > 0 else 100.0
            val01 = (val_raw - min_v) / span
        else:
            # Already in 0-1 scale
            span = (max_v - min_v) if (max_v - min_v) > 0 else 1.0
            val01 = (val_raw - min_v) / span

        # Ensure score is within valid range
        return max(0.0, min(1.0, val01))

    def _score_core(self, context: dict, scorable, dimensions: List[Union[str, dict]]) -> ScoreBundle:
        """
        Score a response against goal text across specified dimensions
        
        Process:
        1. Extract goal and response texts
        2. Compute embeddings for both texts
        3. For each dimension, run model inference
        4. Scale raw outputs to normalized scores
        5. Return comprehensive score bundle
        
        Args:
            context: Contains goal information
            scorable: Response to be scored
            dimensions: List of dimension names or configs to score against
            
        Returns:
            ScoreBundle with results for all dimensions
        """
        # Extract texts for scoring
        goal = context.get(GOAL, {})
        goal_text = goal.get(GOAL_TEXT, "") or ""

        # Compute embeddings once for efficiency
        g_np = self.memory.embedding.get_or_create(goal_text)
        o_np = self.memory.embedding.get_or_create(scorable.text)
        g = torch.tensor(g_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        o = torch.tensor(o_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        results: Dict[str, ScoreResult] = {}

        # Score across all specified dimensions
        for dim in dimensions:
            dim_name = dim.get("name") if isinstance(dim, dict) else dim
            model = self.models.get(dim_name)
            if model is None:
                continue

            # Model inference
            with torch.no_grad():
                z = model.encoder(g, o)  # Encode text pair
                logit = float(model.predictor(z).view(-1)[0].item())  # Raw prediction
                prob = float(torch.sigmoid(torch.tensor(logit)).item())  # Probability

            # Scale to normalized score
            meta = self.model_meta.get(dim_name, {"min_value": 0.0, "max_value": 1.0})
            score01 = self._scale(dim_name, logit, meta)  # Final 0-1 score

            # Build result with comprehensive attributes
            attributes = {
                "q_value": round(logit, 6),  # Raw logit
                "prob": round(prob, 6),      # Sigmoid probability  
                "energy": logit,              # Alternative name for logit
                "min_value": 0.0,            # Normalized range
                "max_value": 1.0,            # Normalized range
            }

            results[dim_name] = ScoreResult(
                dimension=dim_name,
                score=round(score01, 4),     # Final normalized score
                source=self.model_type,
                rationale=f"logit={logit:.4f}, prob={prob:.4f}",  # Debug info
                weight=1.0,
                attributes=attributes,
            )

        return ScoreBundle(results=results)