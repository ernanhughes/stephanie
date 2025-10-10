"""
Output Verification Module for Agentic Tree Search.

This module provides structured verification and scoring of task execution outputs.
It validates multi-dimensional quality metrics against configurable thresholds
and provides detailed feedback on output quality across different dimensions.

The verifier operates on structured result dictionaries containing:
- Overall metric score
- Vector of dimension-specific scores  
- Textual summary
- Merged output content

Key Features:
- Configurable quality thresholds per dimension
- Flexible dimension matching (case-insensitive, partial matches)
- Multiple verification modes (require_all vs any threshold)
- Detailed flag reporting for debugging and analysis
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class OutputVerifier:
    """
    Modern verifier for structured score dictionaries.
    
    Validates expected quality dimensions and enforces configurable thresholds.
    Supports both strict (all thresholds must pass) and lenient (any threshold passes)
    verification modes.
    
    Example usage:
        >>> verifier = OutputVerifier(thresholds={"alignment": 0.8, "clarity": 0.7})
        >>> result = verifier.verify({
        ...     "metric": 0.85,
        ...     "vector": {"alignment_score": 0.9, "clarity_metric": 0.6},
        ...     "summary": "Good alignment but poor clarity"
        ... })
        >>> print(result["is_verified"])  # False - clarity below threshold
    
    Attributes:
        thresholds: Dictionary mapping dimension names to minimum required scores
        require_all: If True, all thresholds must be met; if False, any threshold passing is sufficient
    """

    def __init__(self,
                 thresholds: Optional[Dict[str, float]] = None,
                 require_all: bool = True):
        """
        Initialize the output verifier with quality thresholds.
        
        Args:
            thresholds: Dictionary of dimension_name -> minimum_score pairs.
                       Example: {"alignment": 0.9, "clarity": 0.7, "correctness": 0.8}
            require_all: If True, verification fails if ANY threshold is unmet.
                       If False, verification passes if ANY threshold is met.
                       Default: True (strict mode)
        """
        self.thresholds = thresholds or {"alignment": 0.9}
        self.require_all = require_all

    # --------------------------------------------------------------
    def verify(self,
               result: Dict[str, Any],
               stderr: str = "",
               has_submission_file: bool = False) -> Dict[str, Any]:
        """
        Validate a structured task result against configured quality thresholds.
        
        Processes the result dictionary to extract metrics, validate against
        thresholds, and generate a comprehensive verification report.
        
        Args:
            result: Structured result dictionary containing:
                   - metric: Overall quality score (float)
                   - vector: Dictionary of dimension_name -> score mappings
                   - summary: Human-readable summary of results
                   - merged_output: Combined output content
            stderr: Standard error output from execution (unused in current implementation)
            has_submission_file: Whether a submission file was generated (unused)
            
        Returns:
            Dictionary containing:
            - metric: Verified metric score (may differ from input)
            - summary: Original summary text
            - merged_output: Original merged output
            - is_bug: True if verification failed (output is problematic)
            - is_verified: True if all required thresholds are met
            - flags: Detailed breakdown of each dimension's verification status
            
        Note:
            If input is not a dictionary, returns a default error response with
            is_bug=True and is_verified=False.
        """
        # --- Input validation ---
        if not isinstance(result, dict):
            return {
                "metric": 0.0,
                "summary": "Invalid result type: expected dictionary",
                "is_bug": True,
                "is_verified": False,
                "merged_output": str(result)
            }

        # Extract components from result dictionary with fallbacks
        metric = float(result.get("metric", 0.0))
        vector = result.get("vector", {}) or {}
        summary = result.get("summary", "")
        merged_output = result.get("merged_output", "")

        # --- Threshold validation across all dimensions ---
        flags = {}
        for dimension_name, threshold_value in self.thresholds.items():
            # Find the best matching score for this dimension
            actual_score = self._find_score_for(dimension_name, vector)
            
            # Determine if this dimension meets the threshold
            threshold_met = (actual_score is not None) and (actual_score >= threshold_value)
            
            flags[dimension_name] = {
                "score": actual_score,  # Actual score found (or None)
                "threshold": threshold_value,  # Required minimum score
                "ok": threshold_met  # Whether threshold was met
            }

        # --- Aggregate verification result ---
        # Extract pass/fail status for all checked dimensions
        threshold_results = [flag_data["ok"] for flag_data in flags.values()]
        
        # Apply verification policy: require_all vs require_any
        if self.require_all:
            verified = all(threshold_results)  # All thresholds must pass
        else:
            verified = any(threshold_results)  # Any threshold passing is sufficient

        # --- Compute refined verification metric ---
        # Use average of verified dimension scores if available, fall back to original metric
        verified_scores = [
            flag_data["score"] 
            for flag_data in flags.values() 
            if flag_data["score"] is not None
        ]
        
        if verified_scores:
            # Weighted average or simple mean of passing dimensions
            verify_metric = sum(verified_scores) / len(verified_scores)
        else:
            # Fallback to original metric if no dimension scores available
            verify_metric = metric

        return {
            "metric": verify_metric,
            "summary": summary,
            "merged_output": merged_output,
            "is_bug": not verified,  # True if verification failed
            "is_verified": verified,  # True if quality standards met
            "flags": flags  # Detailed dimension-by-dimension results
        }

    # --------------------------------------------------------------
    def _find_score_for(self, dimension: str, vector: Dict[str, float]) -> Optional[float]:
        """
        Find the most relevant score for a given quality dimension.
        
        Performs flexible matching on dimension names to handle variations
        in scoring key naming conventions. Matching is case-insensitive
        and looks for partial matches.
        
        Examples:
            dimension="alignment" will match keys like:
            - "sicql.alignment"
            - "mrq.alignment_metric" 
            - "Alignment"
            - "overall_alignment"
            
        Args:
            dimension: The quality dimension to find a score for
            vector: Dictionary of score_key -> score_value pairs
            
        Returns:
            The matching score value as float, or None if no match found
        """
        dimension_lower = dimension.lower()
        
        for score_key, score_value in vector.items():
            # Case-insensitive partial matching
            if dimension_lower in score_key.lower():
                try:
                    return float(score_value)
                except (TypeError, ValueError):
                    # Skip non-numeric values that matched by key
                    continue
                    
        return None  # No matching score found