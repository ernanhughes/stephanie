# stephanie/components/gap/processors/analysis.py
from __future__ import annotations
import logging
from typing import Any, Dict
import numpy as np

from ..models import GapConfig


logger = logging.getLogger(__name__)


class AnalysisProcessor:
    """Handles analytical computations like delta analysis and intensity reports."""
    
    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger
    
    async def perform_analysis(self, scoring_results: Dict[str, Any],
                            run_id: str, manifest) -> Dict[str, Any]:
        """Perform all analytical computations."""
        zm = self.container.get("zeromodel")
        storage = self.container.get("storage")
        
        results = {}
        
        # Inter-model delta analysis
        delta_analysis = await self._perform_delta_analysis(
            zm, scoring_results, run_id
        )
        results["delta_analysis"] = delta_analysis
        
        # Intensity report
        intensity_report = await self._generate_intensity_report(
            zm, scoring_results, run_id
        )
        results["intensity"] = intensity_report
        
        # PHOS analysis
        phos_analysis = await self._perform_phos_analysis(
            scoring_results, run_id
        )
        results["phos"] = phos_analysis
        
        return results
    
    async def _perform_delta_analysis(self, zm, scoring_results: Dict[str, Any],
                                    run_id: str) -> Dict[str, Any]:
        """Perform inter-model delta analysis."""
        hrm_matrix = scoring_results["hrm_vectors"]
        tiny_matrix = scoring_results["tiny_vectors"]
        hrm_names = scoring_results["hrm_names"]
        tiny_names = scoring_results["tiny_names"]
        
        delta_meta = zm.render_intermodel_delta(
            hrm_matrix, tiny_matrix,
            names_A=hrm_names,
            names_B=tiny_names,
            output_dir=str(self.config.base_dir / run_id / "metrics"),
            pos_label="HRM",
            neg_label="Tiny",
        )
        
        return delta_meta
    
    async def _generate_intensity_report(self, zm, scoring_results: Dict[str, Any],
                                       run_id: str) -> Dict[str, Any]:
        """Generate intensity report."""
        hrm_matrix = scoring_results["hrm_vectors"]
        tiny_matrix = scoring_results["tiny_vectors"]
        hrm_names = scoring_results["hrm_names"]
        tiny_names = scoring_results["tiny_names"]
        
        intensity = zm.build_intensity_report(
            hrm_matrix=hrm_matrix,
            tiny_matrix=tiny_matrix,
            hrm_metric_names=hrm_names,
            tiny_metric_names=tiny_names,
            out_dir=str(self.config.base_dir / run_id / "metrics"),
            top_k=25,
        )
        
        return intensity
    
    async def _perform_phos_analysis(self, scoring_results: Dict[str, Any],
                                   run_id: str) -> Dict[str, Any]:
        """Perform PHOS-guarded analysis."""
        # This would implement your build_hrm_vs_tiny_guarded logic
        # Simplified version - you'd want to expand this
        try:
            from stephanie.zeromodel.vpm_phos import build_hrm_vs_tiny_guarded
            
            # You'd need to reconstruct the DataFrame from scoring results
            # This is a placeholder for the actual implementation
            df_proj = self._prepare_phos_data(scoring_results, run_id)
            
            phos_res = build_hrm_vs_tiny_guarded(
                df_proj,
                dimensions=self.config.dimensions,
                out_prefix=str(self.config.base_dir / run_id / "visuals" / "vpm"),
                tl_fracs=(0.25, 0.16, 0.36, 0.09),
                delta=0.02,
                interleave=self.config.interleave,
                weights=None,
            )
            
            return phos_res
        except Exception as e:
            self.logger.log("PHOSAnalysisError", {"error": str(e)})
            return {"error": str(e)}
    
    def _prepare_phos_data(self, scoring_results: Dict[str, Any], run_id: str):
        """Prepare data for PHOS analysis."""
        # Implementation would convert scoring results to the expected DataFrame format
        # This is simplified - you'd need to adapt based on your actual data structure
        import pandas as pd
        
        # Placeholder - implement based on your _project_dimensions logic
        return pd.DataFrame()