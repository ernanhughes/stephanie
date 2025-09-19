# stephanie/services/lfl_reward_service.py
from __future__ import annotations

import json
import math
import traceback
from typing import Any, Dict, List, Optional, Tuple


class LFLRewardService:
    """
    Combines VoiceScore (VS) and RepresentationScore (RS) to an LfL reward.
    Expect `metrics` to contain your existing coverage/faithfulness/structure/hallucination.
    
    Features:
    - Configurable weights for representation score components
    - Proper HRM normalization
    - Detailed metrics tracking
    - Graceful fallbacks for missing metrics
    - Configurable LfL weights for VS and RS
    - Support for both normalized and raw metrics
    """
    def __init__(self, logger, config: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.config = config or {
            "representation_weights": {
                "coverage": 0.35,
                "hallucination": 0.25,
                "structure": 0.20,
                "faithfulness": 0.20,
                "hrm": 0.20
            },
            "vs_weights": {
                "vs1": 0.5,
                "vs2": 0.3,
                "vs3": 0.2
            },
            "lf_l_weights": {
                "vs": 0.6,
                "rs": 0.4
            },
            "hrm_min": 0.0,
            "hrm_max": 1.0
        }
        
        # Ensure weights sum to 1.0 for representation_weights
        rep_weights = self.config["representation_weights"]
        total_rep = sum(rep_weights.values())
        if total_rep > 0:
            for k in rep_weights:
                rep_weights[k] /= total_rep
                
        # Ensure weights sum to 1.0 for vs_weights
        vs_weights = self.config["vs_weights"]
        total_vs = sum(vs_weights.values())
        if total_vs > 0:
            for k in vs_weights:
                vs_weights[k] /= total_vs
                
        # Ensure vs and rs weights sum to 1.0
        lfl_weights = self.config["lf_l_weights"]
        total_lfl = sum(lfl_weights.values())
        if total_lfl > 0:
            for k in lfl_weights:
                lfl_weights[k] /= total_lfl

    def representation_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate RepresentationScore (RS) from metrics.
        
        Args:
            metrics: Dictionary containing coverage, hallucination, structure, etc.
            
        Returns:
            RepresentationScore between 0.0 and 1.0
        """
        try:
            # Default values for missing metrics
            coverage = float(metrics.get("claim_coverage", 0.0))
            hallucination = float(metrics.get("hallucination_rate", 1.0))
            structure = float(metrics.get("structure", 0.0))
            faithfulness = float(metrics.get("faithfulness", 0.0))
            hrm = metrics.get("HRM_norm", None)
            
            # Calculate each component
            cov_score = coverage
            hall_score = 1.0 - hallucination
            struct_score = structure
            faith_score = faithfulness
            hrm_score = 0.5
            
            # Normalize HRM if available
            if hrm is not None:
                hrm_min = self.config.get("hrm_min", 0.0)
                hrm_max = self.config.get("hrm_max", 1.0)
                hrm_score = (hrm - hrm_min) / max(1e-8, hrm_max - hrm_min)
                hrm_score = max(0.0, min(1.0, hrm_score))
            
            # Apply weights
            weights = self.config["representation_weights"]
            rs = (
                weights["coverage"] * cov_score +
                weights["hallucination"] * hall_score +
                weights["structure"] * struct_score +
                weights["faithfulness"] * faith_score +
                weights["hrm"] * hrm_score
            )
            
            # Clamp to [0, 1]
            rs = max(0.0, min(1.0, rs))
            
            # Log details
            self.logger.log("RepresentationScoreCalculated", {
                "coverage": cov_score,
                "hallucination": hall_score,
                "structure": struct_score,
                "faithfulness": faith_score,
                "hrm": hrm_score,
                "rs": rs,
                "weights": weights
            })
            
            return rs
            
        except Exception as e:
            self.logger.log("RepresentationScoreError", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metrics": json.dumps(metrics)
            })
            return 0.5  # Default neutral score on error

    def voice_score(self, vs_metrics: Dict[str, float]) -> float:
        """
        Calculate VoiceScore (VS) from individual components.
        
        Args:
            vs_metrics: Dictionary containing VS1, VS2, VS3 components
            
        Returns:
            VoiceScore between 0.0 and 1.0
        """
        try:
            vs1 = float(vs_metrics.get("VS1_embed", 0.5))
            vs2 = float(vs_metrics.get("VS2_style", 0.5))
            vs3 = float(vs_metrics.get("VS3_moves", 0.5))
            
            # Apply weights
            weights = self.config["vs_weights"]
            vs = (
                weights["vs1"] * vs1 +
                weights["vs2"] * vs2 +
                weights["vs3"] * vs3
            )
            
            # Clamp to [0, 1]
            vs = max(0.0, min(1.0, vs))
            
            # Log details
            self.logger.log("VoiceScoreCalculated", {
                "VS1_embed": vs1,
                "VS2_style": vs2,
                "VS3_moves": vs3,
                "vs": vs,
                "weights": weights
            })
            
            return vs
            
        except Exception as e:
            self.logger.log("VoiceScoreError", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "vs_metrics": json.dumps(vs_metrics)
            })
            return 0.5  # Default neutral score on error

    def lfl(self, vs: float, rs: float) -> float:
        """
        Calculate Learning from Learning (LfL) score by combining VS and RS.
        
        Args:
            vs: VoiceScore between 0.0 and 1.0
            rs: RepresentationScore between 0.0 and 1.0
            
        Returns:
            LfL score between 0.0 and 1.0
        """
        try:
            # Apply LfL weights
            weights = self.config["lf_l_weights"]
            lfl = (
                weights["vs"] * vs +
                weights["rs"] * rs
            )
            
            # Clamp to [0, 1]
            lfl = max(0.0, min(1.0, lfl))
            
            # Log details
            self.logger.log("LfLScoreCalculated", {
                "vs": vs,
                "rs": rs,
                "lfl": lfl,
                "weights": weights
            })
            
            return lfl
            
        except Exception as e:
            self.logger.log("LfLScoreError", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "vs": vs,
                "rs": rs
            })
            return 0.5  # Default neutral score on error

    def calculate_lfl(self, vs_metrics: Dict[str, float], metrics: Dict[str, float]) -> float:
        """
        Convenience method to calculate LfL score directly from metrics.
        
        Args:
            vs_metrics: VoiceScore components
            metrics: RepresentationScore metrics
            
        Returns:
            LfL score between 0.0 and 1.0
        """
        vs = self.voice_score(vs_metrics)
        rs = self.representation_score(metrics)
        return self.lfl(vs, rs)