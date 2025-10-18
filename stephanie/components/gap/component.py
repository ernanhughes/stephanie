# stephanie/components/gap/component.py
from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any, Dict
from pathlib import Path

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.gap.models import GapConfig
from stephanie.components.gap.orchestrator import GapAnalysisOrchestrator

_logger = logging.getLogger(__name__)

class GapAgent(BaseAgent):
    """
    GAP Analysis Component: Compare HRM vs Tiny models through multiple metrics
    including timelines, frontier maps, delta analysis, and calibration.
    """
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self._config = self._load_config(cfg)
        self._orchestrator = GapAnalysisOrchestrator(self._config, container, logger, memory=memory)
    
    def _load_config(self, raw_config: Dict[str, Any]) -> GapConfig:
        """Convert raw config to typed GapConfig."""
        return GapConfig(
            dimensions=list(raw_config.get(
                "dimensions", 
                ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]
            )),
            hrm_scorers=list(raw_config.get("hrm_scorers", ["hrm"])),
            tiny_scorers=list(raw_config.get("tiny_scorers", ["tiny"])),
            out_dir=Path(raw_config.get("out_dir", "data/gap_runs/vpm")),
            base_dir=Path(raw_config.get("gap_base_dir", "data/gap_runs")),
            interleave=bool(raw_config.get("interleave", False)),
            progress_log_every=int(raw_config.get("progress_log_every", 25)),
            dedupe_policy=raw_config.get("dedupe_policy", "first_wins"),
<<<<<<< HEAD
            per_dim_cap=raw_config.get("per_dim_cap", 100),
=======
            # per_dim_cap=raw_config.get("per_dim_cap", 1000), # CAP count limit per dimension
            per_dim_cap=raw_config.get("per_dim_cap", 100), 
>>>>>>> main
        )
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute GAP analysis comparing HRM vs Tiny models.
        
        Args:
            context: Execution context containing dataset and run information
            
        Returns:
            Dictionary containing all analysis results and artifact paths
        """
        try:
            result = await self._orchestrator.execute_analysis(context)
            context[self.output_key] = result
            return context
        except Exception as e:
            self.logger.log("GapAnalysisError", {
                "error": str(e),
                "run_id": context.get("pipeline_run_id", "unknown")
            })
            raise