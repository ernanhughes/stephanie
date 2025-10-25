# stephanie/components/gap/gap.py
from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.gap.models import GapConfig
from stephanie.components.gap.orchestrator import GapAnalysisOrchestrator
from stephanie.components.gap.models import GapConfig, EgConfig, EgBadgeConfig, EgRenderConfig, EgThresholds, EgStreams, EgMemConfig, EgModelConfig, EgBaselineConfig

_logger = logging.getLogger(__name__)



class GapAgent(BaseAgent):
    """
    GAP (Gap Analysis Project) Agent - Entry point for comparative model analysis.
    
    This agent orchestrates the comparison between HRM (Hierarchical Reasoning Model) 
    and Tiny models across multiple reasoning dimensions. It serves as the main interface
    for the pipeline to initiate gap analysis, which includes:
    
    - Timeline visualization of model behaviors
    - Frontier maps showing systematic differences
    - Delta (Δ) analysis quantifying model divergence
    - Calibration and routing recommendations
    
    The agent transforms raw model outputs into visual and quantitative insights
    about where and why models disagree, enabling targeted improvements and 
    intelligent model routing.
    
    Configurable Dimensions:
        reasoning, knowledge, clarity, faithfulness, coverage
        
    Example Usage:
        ```python
        agent = GapAgent(cfg, memory, container, logger)
        result = await agent.run(context)
        # Returns: {
        #   "gap": {
        #     "run_id": "gap_123...",
        #     "summary_path": "path/to/summary.json",
        #     "timeline_gif": "path/to/timeline.gif",
        #     "delta_analysis": {...}
        #   }
        # }
        ```
    
    Key Outputs:
        - Visual timelines (VPM/PHOS packs)
        - Frontier comparison maps
        - Δ-maps (HRM - Tiny difference fields)
        - Statistical significance reports
        - Calibration thresholds for routing
    """
    
    def __init__(self, cfg, memory, container, logger):
        """
        Initialize GAP analysis agent with configuration and dependencies.
        
        Args:
            cfg: Agent configuration dictionary
            memory: Memory interface for data access
            container: Dependency injection container
            logger: Structured logging interface
            
        Initializes:
            - Typed configuration (GapConfig)
            - Analysis orchestrator
            - Resource management
        """
        super().__init__(cfg, memory, container, logger)
        self._config = self._load_config(cfg)
        self._orchestrator = GapAnalysisOrchestrator(self._config, container, logger, memory=memory)
    
    def _load_config(self, raw_config: Dict[str, Any]) -> GapConfig:
        """
        Convert raw configuration to typed GapConfig with safe defaults.
        
        Args:
            raw_config: Raw configuration dictionary from pipeline
            
        Returns:
            GapConfig: Typed configuration with validated defaults
            
        Note:
            - Dimensions default to all five reasoning facets
            - Output directories are created if they don't exist
            - per_dim_cap controls dataset size for manageable runs
        """
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
            # Lower cap for development/demo; increase for production analysis
            per_dim_cap=raw_config.get("per_dim_cap", 100), 
            eg=_merge_eg(raw_config),  # ← NEW
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full GAP analysis pipeline comparing HRM vs Tiny models.
        
        This is the main entry point called by the Stephanie pipeline. It:
        1. Validates input context and configuration
        2. Delegates to orchestrator for actual analysis execution
        3. Handles errors with structured logging
        4. Returns results under the "gap" key for pipeline consumption
        
        Args:
            context: Pipeline execution context containing:
                - pipeline_run_id: Unique identifier for this run
                - dataset: Conversation turns to analyze
                - Any additional pipeline-specific parameters
                
        Returns:
            Dictionary containing analysis results under "gap" key:
            {
                "gap": {
                    "run_id": str,                    # Unique GAP run identifier
                    "summary_path": str,              # Path to analysis summary
                    "timeline_gif": str,              # Path to visual timeline
                    "delta_analysis": Dict,           # Statistical differences
                    "routing_thresholds": Dict,       # Recommended calibration
                    "topology_report": Dict           # Persistent homology results
                }
            }
            
        Raises:
            GapAnalysisError: If analysis fails at orchestrator level
            ConfigurationError: If required context parameters are missing
            
        Logs:
            - "GapAnalysisStarted": When analysis begins with run context
            - "GapAnalysisCompleted": On successful completion with result stats
            - "GapAnalysisError": On failure with error details
        """
        try:
            # Log analysis start with context for audit trail
            self.logger.log("GapAnalysisStarted", {
                "run_id": context.get("pipeline_run_id", "unknown"),
                "dimensions": self._config.dimensions,
                "hrm_scorers": self._config.hrm_scorers,
                "tiny_scorers": self._config.tiny_scorers
            })
            
            # Delegate actual analysis execution to orchestrator
            result = await self._orchestrator.execute_analysis(context)
            
            # Structure results for pipeline consumption
            context[self.output_key] = result
            
            # Log successful completion
            self.logger.log("GapAnalysisCompleted", {
                "run_id": context.get("pipeline_run_id", "unknown"),
                "result_keys": list(result.keys()) if result else [],
                "artifacts_generated": len(result.get("artifacts", [])) if result else 0
            })
            
            return context
            
        except Exception as e:
            # Structured error logging for monitoring and debugging
            self.logger.log("GapAnalysisError", {
                "error": str(e),
                "run_id": context.get("pipeline_run_id", "unknown"),
                "config": asdict(self._config) if hasattr(self, '_config') else None
            })
            raise

def _merge_eg(raw: dict) -> EgConfig:
    eg = raw.get("eg", {}) or {}
    # nested dicts → typed dataclasses with safe defaults
    return EgConfig(
        enabled=eg.get("enabled", True),
        badge=EgBadgeConfig(**eg.get("badge", {})),
        render=EgRenderConfig(
            **{**asdict(EgRenderConfig()), **eg.get("render", {})}
        ),
        thresholds=EgThresholds(**eg.get("thresholds", {})),
        streams=EgStreams(**eg.get("streams", {})),
        mem=EgMemConfig(**eg.get("mem", {})),
        models=EgModelConfig(
            **{**asdict(EgModelConfig()), **eg.get("models", {})}
        ),
        baseline=EgBaselineConfig(**eg.get("baseline", {})),
    )

