# stephanie/components/gap/processors/visualization.py
from __future__ import annotations
import logging
from typing import Any, Dict
import numpy as np

from ..models import GapConfig


logger = logging.getLogger(__name__)


class VisualizationProcessor:
    """Handles generation of visual artifacts like GIFs, frontier maps, and heatmaps."""
    
    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger
    
    async def generate_visualizations(self, scoring_results: Dict[str, Any], 
                                   run_id: str, manifest) -> Dict[str, Any]:
        """Generate all visual artifacts."""
        zm = self.container.get("zeromodel")
        storage = self.container.get("storage")  # Assuming storage is available
        
        results = {}
        
        # Generate frontier map
        frontier_results = await self._generate_frontier_map(
            zm, scoring_results, run_id
        )
        results["frontier"] = frontier_results
        
        # Generate delta heatmap
        delta_heatmap = await self._generate_delta_heatmap(
            scoring_results, run_id
        )
        results["delta_heatmap"] = delta_heatmap
        
        # Handle GIF artifacts
        gif_results = await self._process_timeline_gifs(
            scoring_results, run_id, storage
        )
        results["timelines"] = gif_results
        
        return results
    
    async def _generate_frontier_map(self, zm, scoring_results: Dict[str, Any], 
                                   run_id: str) -> Dict[str, Any]:
        """Generate frontier map comparing HRM vs Tiny."""
        hrm_matrix = scoring_results["hrm_vectors"]
        tiny_matrix = scoring_results["tiny_vectors"]
        
        frontier_meta = zm.render_frontier_map(
            hrm_matrix, tiny_matrix,
            out_dir=str(self.config.base_dir / run_id / "visuals"),
            pos_label="HRM",
            neg_label="Tiny",
            k_latent=20
        )
        
        return frontier_meta
    
    async def _generate_delta_heatmap(self, scoring_results: Dict[str, Any], 
                                    run_id: str) -> Dict[str, Any]:
        """Generate absolute difference heatmap."""
        import matplotlib.pyplot as plt
        
        hrm_matrix = scoring_results["hrm_vectors"]
        tiny_matrix = scoring_results["tiny_vectors"]
        
        try:
            delta = hrm_matrix - tiny_matrix
            delta_abs = np.abs(delta)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(delta_abs, cmap="gray", aspect="auto")
            plt.title("|HRM − Tiny| (aligned)")
            plt.axis("off")
            
            heatmap_path = self.config.base_dir / run_id / "visuals" / "delta_heat.png"
            plt.savefig(heatmap_path, dpi=160, bbox_inches="tight")
            plt.close()
            
            return {"path": str(heatmap_path), "status": "success"}
        except Exception as e:
            self.logger.log("DeltaHeatmapError", {"error": str(e)})
            return {"path": None, "status": "error", "error": str(e)}
    
    async def _process_timeline_gifs(self, scoring_results: Dict[str, Any],
                                   run_id: str, storage) -> Dict[str, Any]:
        """Process and store timeline GIFs."""
        hrm_gif_source = scoring_results.get("hrm_gif")
        tiny_gif_source = scoring_results.get("tiny_gif")
        
        # Extract paths from worker results (handling different return formats)
        def extract_gif_path(gif_result):
            if isinstance(gif_result, dict):
                return gif_result.get("output_path")
            return gif_result
        
        hrm_gif_path = storage.copy_visual_artifact(
            extract_gif_path(hrm_gif_source), run_id, "hrm_timeline.gif"
        )
        tiny_gif_path = storage.copy_visual_artifact(
            extract_gif_path(tiny_gif_source), run_id, "tiny_timeline.gif"
        )
        
        return {
            "hrm_gif": str(hrm_gif_path),
            "tiny_gif": str(tiny_gif_path)
        }