# stephanie/metrics.py
import logging
from typing import Dict, Optional, Union

import numpy as np
import torch

from stephanie.logs.json_logger import JSONLogger


class EpistemicMetrics:
    """
    Class for tracking and analyzing epistemic uncertainty in Stephanie's scoring system.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or JSONLogger("metrics")
    
    @staticmethod
    def compute_uncertainty(
        q_values: Union[np.ndarray, torch.Tensor],
        v_values: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Calculate epistemic uncertainty using Q-V gap.
        
        Args:
            q_values: Predicted Q-values from the model
            v_values: Estimated state values from the V-head
            
        Returns:
            Array of uncertainty values
        """
        # Convert tensors to numpy arrays
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.detach().cpu().numpy()
        if isinstance(v_values, torch.Tensor):
            v_values = v_values.detach().cpu().numpy()
            
        # Ensure shapes match
        if q_values.shape != v_values.shape:
            raise ValueError(f"Shape mismatch: Q({q_values.shape}) vs V({v_values.shape})")
            
        # Calculate absolute difference
        return np.abs(q_values - v_values)
    
    def log_epistemic_gap(
        self,
        gap_info: Dict[str, Union[int, float, str]],
        threshold: float = 0.3
    ) -> None:
        """
        Log and analyze epistemic gaps in model predictions.
        
        Args:
            gap_info: Dictionary containing gap information
            threshold: Threshold for flagging significant gaps
        """
        try:
            uncertainty = gap_info.get("uncertainty", 0.0)
            gap_info["is_significant"] = uncertainty > threshold
            
            # Log to JSON logger
            self.logger.log("EpistemicGap", gap_info)
            
            # Print warning for significant gaps
            if gap_info["is_significant"]:
                self._log_significant_gap(gap_info)
                
        except Exception as e:
            self.logger.log("EpistemicGapError", {
                "error": str(e),
                "gap_info": gap_info
            })
    
    def _log_significant_gap(self, gap_info: dict) -> None:
        """Handle logging for significant epistemic gaps"""
        self.logger.warning(
            f"High epistemic uncertainty detected in {gap_info['dimension']} "
            f"(ID: {gap_info['document_id']}): {gap_info['uncertainty']:.2f}\n"
            f"LLM score: {gap_info['llm_score']:.2f}, "
            f"Model score: {gap_info['predicted_score']:.2f}"
        )
        
        # Add to retraining queue if configured
        if hasattr(self, 'retraining_queue'):
            self.retraining_queue.add(
                dimension=gap_info['dimension'],
                document_id=gap_info['document_id'],
                uncertainty=gap_info['uncertainty']
            )

# Standalone functions for compatibility
def compute_uncertainty(q_values, v_values):
    """Legacy function for backward compatibility"""
    return EpistemicMetrics.compute_uncertainty(q_values, v_values)

def log_epistemic_gap(gap_info, logger=None):
    """Legacy function for backward compatibility"""
    metrics = EpistemicMetrics(logger=logger)
    metrics.log_epistemic_gap(gap_info)