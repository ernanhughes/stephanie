# stephanie/components/gap/processors/calibration.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from stephanie.components.gap.models import GapConfig


logger = logging.getLogger(__name__)


class CalibrationProcessor:
    """Handles calibration and routing analysis between HRM and Tiny models."""
    
    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger
    
    async def analyze_calibration(self, scoring_results: Dict[str, Any],
                                run_id: str, manifest) -> Dict[str, Any]:
        """Perform calibration analysis and routing simulation."""
        storage = self.container.get("storage")
        
        # Load aligned matrices
        hrm_matrix, hrm_names = storage.load_matrix(run_id, "hrm")
        tiny_matrix, tiny_names = storage.load_matrix(run_id, "tiny")

        # keep these around and pass to _simulate_routing:
        results = {}
        
        # Dimension calibration
        calibration_results = await self._calibrate_dimensions(
            hrm_matrix, tiny_matrix, hrm_names, tiny_names, run_id
        )
        results["calibration"] = calibration_results
        
        # Routing simulation
        routing_results = await self._simulate_routing(
            hrm_matrix, tiny_matrix, hrm_names, tiny_names, 
            calibration_results, run_id
        )
        
        results["routing"] = routing_results
        
        return results
    
    async def _calibrate_dimensions(self, hrm_matrix: np.ndarray, tiny_matrix: np.ndarray,
                                  hrm_names: List[str], tiny_names: List[str],
                                  run_id: str) -> Dict[str, Any]:
        """Calibrate Tiny scores to HRM reference for each dimension."""
        calib_params = {}
        dim_stats = []
        
        for dim in self.config.dimensions:
            hrm_col = self._find_dimension_column(hrm_names, "hrm", dim)
            tiny_col = self._find_dimension_column(tiny_names, "tiny", dim)
            
            if hrm_col is None or tiny_col is None:
                dim_stats.append({
                    "dimension": dim, "status": "missing_columns",
                    "hrm_col": hrm_col, "tiny_col": tiny_col
                })
                continue
            
            hrm_scores = self._safe_clip(hrm_matrix[:, hrm_col])
            tiny_scores = self._safe_clip(tiny_matrix[:, tiny_col])
            
            # Fit calibration
            calib = self._monotone_calibration(tiny_scores, hrm_scores)
            calib_params[dim] = calib
            
            # Evaluate calibration
            mae_pre = self._mean_absolute_error(tiny_scores, hrm_scores)
            tiny_calibrated = self._apply_calibration(tiny_scores, calib)
            mae_post = self._mean_absolute_error(tiny_calibrated, hrm_scores)
            
            dim_stats.append({
                "dimension": dim,
                "status": "calibrated",
                "mae_pre": float(mae_pre),
                "mae_post": float(mae_post),
                "improvement": float(mae_pre - mae_post),
                "hrm_col": hrm_col,
                "tiny_col": tiny_col
            })
        
        # Save calibration parameters
        storage = self.container.get("storage")
        calibration_path = storage.save_metrics_report(
            {"parameters": calib_params, "statistics": dim_stats},
            run_id, "calibration_parameters"
        )
        
        return {
            "parameters": calib_params,
            "statistics": dim_stats,
            "path": str(calibration_path)
        }
    
    async def _simulate_routing(self, hrm_matrix: np.ndarray, tiny_matrix: np.ndarray,
                            hrm_names: List[str], tiny_names: List[str],
                            calibration_results: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Simulate routing based on Tiny diagnostics and calibration."""
        # Extract diagnostic signals
        uncertainty_scores = self._extract_diagnostic_signal(tiny_matrix, tiny_names, ("uncertainty",))
        ood_scores = self._extract_diagnostic_signal(tiny_matrix, tiny_names, ("ood_hat",))

        # Create routing mask
        uncertainty_threshold = self.config.route_threshold_uncertainty
        ood_threshold = self.config.route_threshold_ood
        
        use_hrm_mask = (uncertainty_scores > uncertainty_threshold) | (ood_scores > ood_threshold)
        usage_rate = float(np.mean(use_hrm_mask))
        
        # Simulate routed scores for each dimension
        per_dim_results = []
        calib_params = calibration_results["parameters"]
        
        for dim in self.config.dimensions:
            if dim not in calib_params:
                per_dim_results.append({"dimension": dim, "status": "skipped"})
                continue

            hrm_col = self._find_dimension_column(hrm_names, "hrm", dim)
            tiny_col = self._find_dimension_column(tiny_names, "tiny", dim)

            if hrm_col is None or tiny_col is None:
                per_dim_results.append({"dimension": dim, "status": "missing_columns"})
                continue

            hrm_scores = self._safe_clip(hrm_matrix[:, hrm_col])
            tiny_scores = self._safe_clip(tiny_matrix[:, tiny_col])
            tiny_calibrated = self._apply_calibration(tiny_scores, calib_params[dim])

            final_scores = np.where(use_hrm_mask, hrm_scores, tiny_calibrated)
            
            # Compute metrics
            mae_routed = self._mean_absolute_error(final_scores, hrm_scores)
            mae_tiny = self._mean_absolute_error(tiny_scores, hrm_scores)
            mae_calibrated = self._mean_absolute_error(tiny_calibrated, hrm_scores)
            
            per_dim_results.append({
                "dimension": dim,
                "status": "routed",
                "mae_routed": float(mae_routed),
                "mae_tiny": float(mae_tiny),
                "mae_calibrated": float(mae_calibrated),
                "routing_usage": float(np.mean(use_hrm_mask))
            })
        
        # Save routing results
        storage = self.container.get("storage")
        routing_path = storage.save_metrics_report(
            {
                "usage_rate": usage_rate,
                "thresholds": {
                    "uncertainty": uncertainty_threshold,
                    "ood": ood_threshold
                },
                "per_dimension_results": per_dim_results
            },
            run_id, "routing_simulation"
        )
        
        return {
            "usage_rate": usage_rate,
            "per_dimension_results": per_dim_results,
            "path": str(routing_path)
        }
    
    def _find_dimension_column(self, names: List[str], model: str, dimension: str) -> Optional[int]:
        """Find column index for a dimension score."""
        candidates = [
            f"{model}.{dimension}.score",
            f"{model}.{dimension}.aggregate", 
            f"{model}.{dimension}",
        ]
        
        for candidate in candidates:
            if candidate in names:
                return names.index(candidate)
        
        # Fallback: loose match
        for i, name in enumerate(names):
            if (name.startswith(f"{model}.{dimension}") and 
                any(name.endswith(suffix) for suffix in [".score", ".aggregate", ".value", ".raw"])):
                return i
        
        return None
    
    def _extract_diagnostic_signal(self, matrix, names, key_parts):
        for i, name in enumerate(names):
            if name.startswith("tiny.") and all(part in name for part in key_parts):
                return self._safe_clip(matrix[:, i])
        # fallback: zeros (same length as rows)
        return np.zeros(matrix.shape[0], dtype=np.float64)

