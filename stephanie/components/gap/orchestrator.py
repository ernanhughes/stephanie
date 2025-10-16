# stephanie/components/gap/orchestrator.py

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List
from dataclasses import asdict

from .models import GapConfig, GapRunManifest
from .processors.scoring import ScoringProcessor
from .processors.visualization import VisualizationProcessor
from .processors.analysis import AnalysisProcessor
from .processors.calibration import CalibrationProcessor
from .io.manifest import ManifestManager
from .io.storage import GapStorage

_logger = logging.getLogger(__name__)

class GapAnalysisOrchestrator:
    """Orchestrates the complete GAP analysis workflow."""
    
    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger
        
        # Initialize processors
    
    async def execute_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete GAP analysis pipeline."""
        run_id = context["pipeline_run_id"]
        dataset = context.get("dataset", "unknown")
        
        # Phase 1: Setup and sample preparation
        manifest = await self._setup_analysis(run_id, dataset)
        
        # Phase 2: Sample collection and deduplication
        triples_data = await self.scoring_processor.prepare_samples(
            self.config.dimensions, self.memory
        )
        
        # Phase 3: Scoring and timeline generation
        scoring_results = await self.scoring_processor.execute_scoring(
            triples_data, run_id, manifest
        )
        
        # Phase 4: Visualization and analysis
        viz_results = await self.viz_processor.generate_visualizations(
            scoring_results, run_id, manifest
        )
        
        analysis_results = await self.analysis_processor.perform_analysis(
            scoring_results, run_id, manifest
        )
        
        # Phase 5: Calibration and routing
        calibration_results = await self.calibration_processor.analyze_calibration(
            scoring_results, run_id, manifest
        )
        
        # Finalize and return results
        return await self._finalize_analysis(
            manifest, scoring_results, viz_results, 
            analysis_results, calibration_results
        )
    
    async def _setup_analysis(self, run_id: str, dataset: str) -> GapRunManifest:
        """Initialize analysis run with manifest and directory structure."""
        models = {
            "hrm": self.config.hrm_scorers[0],
            "tiny": self.config.tiny_scorers[0]
        }
        
        manifest = GapRunManifest(
            run_id=run_id,
            dataset=dataset,
            models=models,
            dimensions=self.config.dimensions
        )
        
        await self.manifest_manager.initialize_run(manifest)
        return manifest
    
    async def _finalize_analysis(self, manifest: GapRunManifest, 
                               scoring_results: Dict[str, Any],
                               viz_results: Dict[str, Any], 
                               analysis_results: Dict[str, Any],
                               calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all results and update manifest."""
        combined_results = {
            "scoring": scoring_results,
            "visualization": viz_results,
            "analysis": analysis_results,
            "calibration": calibration_results,
            "manifest": asdict(manifest)
        }
        
        await self.manifest_manager.finalize_run(manifest, combined_results)
        return combined_results