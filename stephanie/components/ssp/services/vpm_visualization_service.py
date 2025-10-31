# stephanie/components/ssp/services/vpm_visualization_service.py
"""
VPM Visualization Service - Generates Vectorized Performance Map images for SSP episodes

This service converts SSP episode data into VPM visualizations that help track:
- Performance trends across multiple dimensions
- Difficulty progression in the curriculum
- Evidence quality and solver efficiency
- Verification success patterns

The service uses the PHOS (Packed High-Order Structure) algorithm to create 
meaningful visual representations of multi-dimensional performance metrics.

Key features:
- Generates both raw VPM and PHOS-packed visualizations
- Creates comparison visualizations between different SSP configurations
- Tracks historical trends for curriculum management
- Integrates with the VPM control system for decision support
- Configurable visualization styles and parameters
"""

from __future__ import annotations

import os
import time
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from stephanie.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.services.service_protocol import Service
from stephanie.components.ssp.utils.trace import EpisodeTrace

# Import VPM visualization components from the examples provided
from stephanie.zeromodel.vpm_phos import (
    build_vpm_phos_artifacts,
    build_compare_guarded,
    robust01,
    brightness_concentration,
    image_entropy,
    save_img,
    vpm_vector_from_df,
    to_square,
    phos_sort_pack
)
from stephanie.zeromodel.vpm_controller import VPMRow, Signal, Thresholds, Policy

class VPMVisualizationService(Service):
    """
    Service for generating VPM visualization artifacts from SSP episode data.
    
    This service:
    - Converts EpisodeTrace objects to VPM-compatible metric formats
    - Generates both raw VPM and PHOS-packed visualizations
    - Creates comparison visualizations between different SSP configurations
    - Tracks historical trends for curriculum management
    - Integrates with the VPM control system for decision support
    
    The service follows a stateful pattern where each processing unit (e.g., question, episode)
    has its own controller instance that tracks its performance history.
    """
    
    def __init__(
        self, 
        cfg: Dict[str, Any], 
        memory: MemoryTool, 
        logger: JSONLogger, 
        container: Optional[Any] = None
    ):
        """
        Initialize the VPM Visualization Service.
        
        Args:
            cfg: Configuration dictionary with VPM parameters
            memory: Memory tool for state persistence
            logger: JSON logger for structured logging
            container: Dependency injection container
        """
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self.container = container
        self._initialized = False

        # Service state
        self._metrics_history: Dict[str, List[Dict[str, float]]] = {}
        self._episode_traces: Dict[str, EpisodeTrace] = {}
        
        # Visualization configuration
        self._setup_visualization_paths()
        
        # Metrics configuration
        self._setup_metrics()
        
        # VPM parameters
        self._setup_vpm_parameters()

    def _setup_visualization_paths(self) -> None:
        """Configure paths for VPM visualization artifacts."""
        c = (self.cfg.get("vpm_viz") or {})
        
        # Base directory for visualization artifacts
        self._viz_dir = c.get("output_dir", "./runs/vpm_visualizations")
        os.makedirs(self._viz_dir, exist_ok=True)
        
        # Subdirectories for different visualization types
        self._raw_viz_dir = os.path.join(self._viz_dir, "raw")
        self._phos_viz_dir = os.path.join(self._viz_dir, "phos")
        self._compare_viz_dir = os.path.join(self._viz_dir, "comparison")
        self._episode_data_dir = os.path.join(self._viz_dir, "episode_data")
        
        os.makedirs(self._raw_viz_dir, exist_ok=True)
        os.makedirs(self._phos_viz_dir, exist_ok=True)
        os.makedirs(self._compare_viz_dir, exist_ok=True)
        os.makedirs(self._episode_data_dir, exist_ok=True)
        
        # TL fractions for PHOS guard sweep
        self._tl_fracs = c.get("tl_fracs", [0.25, 0.16, 0.36, 0.09])
        self._delta = c.get("delta", 0.02)
        
        # Dimensions to visualize
        self._dimensions = c.get("dimensions", [
            "verifier_f1", "difficulty", "steps_norm", "evidence_cnt",
            "coverage", "correctness", "coherence", "citation_support", "entity_consistency"
        ])

    def _setup_metrics(self) -> None:
        """Configure metrics conversion and normalization parameters."""
        # Default metric ranges for normalization
        self._metric_ranges = {
            "verifier_f1": (0.0, 1.0),
            "difficulty": (0.0, 1.0),
            "steps_norm": (0.0, 1.0),
            "evidence_cnt": (0.0, 1.0),
            "coverage": (0.0, 1.0),
            "correctness": (0.0, 1.0),
            "coherence": (0.0, 1.0),
            "citation_support": (0.0, 1.0),
            "entity_consistency": (0.0, 1.0)
        }
        
        # Override with config values if provided
        metric_cfg = self.cfg.get("vpm_viz", {}).get("metric_ranges", {})
        for metric, (min_val, max_val) in metric_cfg.items():
            if metric in self._metric_ranges:
                self._metric_ranges[metric] = (float(min_val), float(max_val))

    def _setup_vpm_parameters(self) -> None:
        """Configure VPM-specific parameters for visualization generation."""
        c = (self.cfg.get("vpm_viz") or {})
        
        # PHOS parameters
        self._phos_tl_frac = float(c.get("phos_tl_frac", 0.25))
        self._phos_interleave = bool(c.get("phos_interleave", False))
        self._phos_weights = c.get("phos_weights", None)
        
        # Raw VPM parameters
        self._raw_vpm_interleave = bool(c.get("raw_vpm_interleave", False))
        self._raw_vpm_weights = c.get("raw_vpm_weights", None)
        
        # Comparison parameters
        self._compare_models = c.get("compare_models", ["current", "baseline"])
        self._compare_tl_fracs = c.get("compare_tl_fracs", self._tl_fracs)
        self._compare_delta = float(c.get("compare_delta", self._delta))

    # ---------------- Service Protocol Implementation ----------------
    
    def initialize(self, **kwargs) -> None:
        """Initialize the service and required resources."""
        if self._initialized:
            return
            
        self._initialized = True
        self.logger.log("VPMVisualizationServiceInit", {
            "viz_dir": self._viz_dir,
            "tl_fracs": self._tl_fracs,
            "delta": self._delta,
            "dimensions": self._dimensions,
            "metric_ranges": self._metric_ranges
        })

    def shutdown(self) -> None:
        """Clean up resources and persist state if needed."""
        # Clear internal caches
        self._metrics_history.clear()
        self._episode_traces.clear()
        
        self._initialized = False
        self.logger.log("VPMVisualizationServiceShutdown", {})

    def health_check(self) -> Dict[str, Any]:
        """Return service health status and metrics."""
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "episode_count": len(self._episode_traces),
            "traced_dimensions": len(self._metrics_history),
            "timestamp": time.time(),
            "viz_dir": self._viz_dir,
            "active_dimensions": self._dimensions
        }

    @property
    def name(self) -> str:
        """Service name for identification."""
        return "ssp-vpm-visualization"

    # ---------------- Public API ----------------
    
    def generate_episode_visualization(
        self, 
        unit: str,
        episode: EpisodeTrace,
        step_idx: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate VPM visualization artifacts for a specific SSP episode.
        
        Args:
            unit: Identifier for the processing unit (e.g., question ID)
            episode: EpisodeTrace object containing the episode data
            step_idx: Current step index in the processing pipeline
            output_path: Custom output path (defaults to configured viz_dir)
            
        Returns:
            Dictionary with paths to generated visualization artifacts
        """
        # Store episode trace for potential comparison
        self._episode_traces[unit] = episode
        
        # Convert episode to VPM row and track metrics
        vpm_row = self._episode_to_vpm_row(unit, episode, step_idx)
        self._track_metrics(unit, vpm_row, step_idx)
        
        # Generate visualization
        return self.generate_visualization(
            unit=unit,
            step_idx=step_idx,
            output_path=output_path
        )

    def generate_visualization(
        self, 
        unit: str, 
        step_idx: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate VPM visualization artifacts for a specific unit.
        
        Args:
            unit: Identifier for the processing unit
            step_idx: Current step index (for naming)
            output_path: Custom output path (defaults to configured viz_dir)
            
        Returns:
            Dictionary with paths to generated visualization artifacts
        """
        # Get metrics history for this unit
        metrics_history = self._metrics_history.get(unit, [])
        if not metrics_history:
            return {}
            
        # Convert to DataFrame format expected by VPM builder
        df = self._convert_to_dataframe(metrics_history)
        
        # Determine output path
        if output_path is None:
            output_path = os.path.join(self._phos_viz_dir, f"{unit.replace(':', '_')}")
            
        # Generate PHOS artifacts
        artifacts = build_vpm_phos_artifacts(
            df,
            model="ssp",
            dimensions=self._dimensions,
            out_prefix=output_path,
            tl_frac=self._phos_tl_frac,
            interleave=self._phos_interleave,
            weights=self._phos_weights,
        )
        
        # Generate raw VPM as well
        raw_output_path = output_path.replace(self._phos_viz_dir, self._raw_viz_dir)
        raw_artifacts = build_vpm_phos_artifacts(
            df,
            model="ssp",
            dimensions=self._dimensions,
            out_prefix=raw_output_path,
            tl_frac=0.0,  # Raw VPM doesn't need PHOS packing
            interleave=self._raw_vpm_interleave,
            weights=self._raw_vpm_weights,
        )
        
        # Save episode data for future reference
        self._save_episode_data(unit, metrics_history, output_path)
        
        return {
            "raw": raw_artifacts["paths"]["raw"],
            "phos": artifacts["paths"]["phos"],
            "metrics": json.dumps(artifacts["metrics"]),
            "episode_data": os.path.join(self._episode_data_dir, f"{unit}.json")
        }

    def generate_comparison_visualization(
        self,
        unit: str,
        model_a: str,
        model_b: str,
        output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate comparison visualization between two models for a unit.
        
        Args:
            unit: Identifier for the processing unit
            model_a: First model identifier
            model_b: Second model identifier
            output_path: Custom output path
            
        Returns:
            Dictionary with paths to generated comparison artifacts
        """
        # Determine output path
        if output_path is None:
            output_path = os.path.join(
                self._compare_viz_dir, 
                f"{unit.replace(':', '_')}_{model_a}_vs_{model_b}"
            )
            
        # Generate comparison artifacts
        artifacts = build_compare_guarded(
            df=self._get_comparison_dataframe(unit, model_a, model_b),
            dimensions=self._dimensions,
            out_prefix=output_path,
            model_A=model_a,
            model_B=model_b,
            tl_fracs=self._compare_tl_fracs,
            delta=self._compare_delta,
            interleave=self._phos_interleave,
            weights=self._phos_weights,
        )
        
        return {
            "summary": artifacts.get("summary", {}),
            "sweep": artifacts.get("sweep", {}),
            "diff_range": artifacts.get("diff_range"),
            "diff_image": f"{output_path}_vpm_chosen_diff.png",
            "model_a_chosen": artifacts["sweep"].get(model_a, [{}])[-1].get("phos_path", ""),
            "model_b_chosen": artifacts["sweep"].get(model_b, [{}])[-1].get("phos_path", "")
        }

    def generate_curriculum_visualization(
        self,
        output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate visualization showing curriculum progression over time.
        
        Args:
            output_path: Custom output path
            
        Returns:
            Dictionary with paths to generated visualization artifacts
        """
        # Aggregate metrics across all units
        all_metrics = []
        for unit, metrics in self._metrics_history.items():
            for metric in metrics:
                all_metrics.append({
                    "unit": unit,
                    **metric
                })
        
        if not all_metrics:
            return {}
            
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Create a special curriculum-focused output path
        if output_path is None:
            timestamp = int(time.time())
            output_path = os.path.join(self._phos_viz_dir, f"curriculum_progression_{timestamp}")
            
        # Generate PHOS artifacts
        artifacts = build_vpm_phos_artifacts(
            df,
            model="curriculum",
            dimensions=self._dimensions,
            out_prefix=output_path,
            tl_frac=self._phos_tl_frac,
            interleave=self._phos_interleave,
            weights=self._phos_weights,
        )
        
        return {
            "curriculum_phos": artifacts["paths"]["phos"],
            "curriculum_raw": artifacts["paths"]["raw"],
            "metrics": json.dumps(artifacts["metrics"]),
        }

    # ---------------- Internal Helpers ----------------
    
    def _episode_to_vpm_row(
        self, 
        unit: str,
        episode: EpisodeTrace, 
        step_idx: Optional[int] = None
    ) -> VPMRow:
        """
        Convert an EpisodeTrace to a VPMRow for visualization.
        
        Args:
            unit: Identifier for the processing unit
            episode: EpisodeTrace object
            step_idx: Current step index
            
        Returns:
            VPMRow object compatible with visualization pipeline
        """
        # Convert episode to metrics
        dims = self.episode_to_dims(episode)
        
        # Create VPMRow
        return VPMRow(
            unit=unit,
            kind="text",  # SSP is primarily text processing
            timestamp=time.time(),
            step_idx=step_idx,
            dims=dims,
            meta={
                "episode_id": episode.episode_id,
                "verified": episode.verified,
                "question": episode.question,
                "predicted_answer": episode.predicted_answer,
                "solver_steps": episode.solver_steps,
                "evidence_count": len(episode.evidence_docs),
                "difficulty": episode.difficulty
            }
        )

    def episode_to_dims(self, ep: EpisodeTrace) -> Dict[str, float]:
        """
        Normalize episode metrics into [0,1] ranges for visualization.
        
        Args:
            ep: EpisodeTrace object
            
        Returns:
            Dictionary of normalized metrics
        """
        # Normalize simple dims into [0,1] where sensible
        dims = {
            "verifier_f1": float(ep.reward),  # already 0..1-ish with F1
            "difficulty": float(ep.difficulty),  # assume 0..1 curriculum
            "steps_norm": min(1.0, ep.solver_steps / 16.0),  # clip to a small bound
            "evidence_cnt": min(1.0, len(ep.evidence_docs) / 8.0),
        }
        
        # Add any additional metrics from meta if present
        if hasattr(ep, 'meta') and isinstance(ep.meta, dict):
            # Coverage - assume it's in the meta
            if 'coverage' in ep.meta:
                dims['coverage'] = min(1.0, max(0.0, float(ep.meta['coverage'])))
            
            # Correctness - assume it's in the meta
            if 'correctness' in ep.meta:
                dims['correctness'] = min(1.0, max(0.0, float(ep.meta['correctness'])))
            
            # Coherence - assume it's in the meta
            if 'coherence' in ep.meta:
                dims['coherence'] = min(1.0, max(0.0, float(ep.meta['coherence'])))
            
            # Citation support - assume it's in the meta
            if 'citation_support' in ep.meta:
                dims['citation_support'] = min(1.0, max(0.0, float(ep.meta['citation_support'])))
            
            # Entity consistency - assume it's in the meta
            if 'entity_consistency' in ep.meta:
                dims['entity_consistency'] = min(1.0, max(0.0, float(ep.meta['entity_consistency'])))
        
        # Ensure all required dimensions have values
        for dim in self._dimensions:
            if dim not in dims:
                dims[dim] = 0.0
                
        return dims

    def _track_metrics(
        self, 
        unit: str, 
        vpm_row: VPMRow, 
        step_idx: Optional[int]
    ) -> None:
        """Track metrics history for visualization purposes."""
        if unit not in self._metrics_history:
            self._metrics_history[unit] = []
            
        # Create a record with step index and metrics
        record = {
            "step_idx": step_idx or len(self._metrics_history[unit]),
            **{k: float(v) for k, v in vpm_row.dims.items()}
        }
        
        self._metrics_history[unit].append(record)
        
        # Keep history bounded
        max_history = self.cfg.get("vpm_viz", {}).get("max_metrics_history", 100)
        if len(self._metrics_history[unit]) > max_history:
            self._metrics_history[unit] = self._metrics_history[unit][-max_history:]

    def _convert_to_dataframe(self, metrics_history: List[Dict]) -> pd.DataFrame:
        """
        Convert metrics history to DataFrame format expected by VPM builder.
        """
        try:
            # Create a DataFrame with multi-index for model and dimension
            df_data = []
            for i, record in enumerate(metrics_history):
                step_idx = record["step_idx"]
                for dim, value in record.items():
                    if dim == "step_idx":
                        continue
                    df_data.append({
                        "node_id": f"{step_idx}",
                        "ssp": value,
                        "dimension": dim
                    })
            
            df = pd.DataFrame(df_data)
            # Pivot to get dimensions as columns
            df = df.pivot(index="node_id", columns="dimension", values="ssp").reset_index()
            
            # Ensure all dimensions are present
            for dim in self._dimensions:
                if dim not in df.columns:
                    df[dim] = 0.0
            
            return df
        except Exception as e:
            self.logger.log("VPMVisualizationError", {
                "event": "dataframe_conversion_failed",
                "error": str(e)
            })
            # Fallback for environments without pandas
            return pd.DataFrame({
                "node_id": [str(i) for i in range(len(metrics_history))],
                **{dim: [0.0] * len(metrics_history) for dim in self._dimensions}
            })

    def _get_comparison_dataframe(
        self, 
        unit: str, 
        model_a: str, 
        model_b: str
    ) -> pd.DataFrame:
        """
        Create a DataFrame suitable for model comparison visualization.
        """
        # Get metrics for both models
        model_a_metrics = self._get_model_metrics(unit, model_a)
        model_b_metrics = self._get_model_metrics(unit, model_b)
        
        # Convert to DataFrames
        df_a = self._convert_to_dataframe(model_a_metrics)
        df_b = self._convert_to_dataframe(model_b_metrics)
        
        # Merge DataFrames
        df = df_a.merge(df_b, on="node_id", suffixes=("_a", "_b"))
        
        # Rename columns to match expected format
        for dim in self._dimensions:
            if f"{dim}_a" in df.columns and f"{dim}_b" in df.columns:
                df[f"{model_a}.{dim}"] = df[f"{dim}_a"]
                df[f"{model_b}.{dim}"] = df[f"{dim}_b"]
                df.drop([f"{dim}_a", f"{dim}_b"], axis=1, inplace=True)
                
        return df

    def _get_model_metrics(self, unit: str, model: str) -> List[Dict]:
        """
        Get metrics for a specific model.
        """
        if model == "current" and unit in self._metrics_history:
            return self._metrics_history[unit]
            
        # For baseline or other models, you would need to implement retrieval logic
        # This is a placeholder for actual implementation
        return []
        
    def _save_episode_data(self, unit: str, metrics_history: List[Dict], output_path: str) -> None:
        """
        Save episode data for future reference and analysis.
        """
        try:
            # Get the latest episode trace
            episode = self._episode_traces.get(unit)
            if not episode:
                return
                
            # Create episode data dictionary
            episode_data = {
                "unit": unit,
                "episode_id": episode.episode_id,
                "seed_answer": episode.seed_answer,
                "question": episode.question,
                "predicted_answer": episode.predicted_answer,
                "verified": episode.verified,
                "reward": episode.reward,
                "difficulty": episode.difficulty,
                "solver_steps": episode.solver_steps,
                "evidence_docs": episode.evidence_docs,
                "meta": episode.meta,
                "metrics_history": metrics_history
            }
            
            # Save to JSON
            data_path = os.path.join(self._episode_data_dir, f"{unit.replace(':', '_')}.json")
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(episode_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.log("VPMVisualizationError", {
                "event": "episode_data_save_failed",
                "unit": unit,
                "error": str(e)
            })

    def generate_raw_vpm_image(
        self, 
        unit: str,
        step_idx: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a raw VPM image (without PHOS packing) for a specific unit.
        
        Args:
            unit: Identifier for the processing unit
            step_idx: Current step index
            output_path: Custom output path
            
        Returns:
            Path to the generated image
        """
        # Get metrics history for this unit
        metrics_history = self._metrics_history.get(unit, [])
        if not metrics_history:
            return ""
            
        # Convert to VPM vector
        df = self._convert_to_dataframe(metrics_history)
        vec = vpm_vector_from_df(
            df,
            model="ssp",
            dimensions=self._dimensions,
            interleave=self._raw_vpm_interleave,
            weights=self._raw_vpm_weights
        )
        
        # Convert to square image
        img, _ = to_square(vec)
        
        # Save image
        if output_path is None:
            output_path = os.path.join(
                self._raw_viz_dir, 
                f"{unit.replace(':', '_')}_raw_vpm_{int(time.time())}.png"
            )
            
        save_img(img, output_path, title=f"SSP VPM (Raw) - {unit}")
        
        return output_path

    def generate_phos_image(
        self, 
        unit: str,
        step_idx: Optional[int] = None,
        output_path: Optional[str] = None,
        tl_frac: Optional[float] = None
    ) -> str:
        """
        Generate a PHOS-packed VPM image for a specific unit.
        
        Args:
            unit: Identifier for the processing unit
            step_idx: Current step index
            output_path: Custom output path
            tl_frac: Top-left fraction for PHOS packing
            
        Returns:
            Path to the generated image
        """
        # Get metrics history for this unit
        metrics_history = self._metrics_history.get(unit, [])
        if not metrics_history:
            return ""
            
        # Convert to VPM vector
        df = self._convert_to_dataframe(metrics_history)
        vec = vpm_vector_from_df(
            df,
            model="ssp",
            dimensions=self._dimensions,
            interleave=self._phos_interleave,
            weights=self._phos_weights
        )
        
        # Apply PHOS packing
        tl_frac = tl_frac if tl_frac is not None else self._phos_tl_frac
        img = phos_sort_pack(vec, tl_frac=tl_frac)
        
        # Save image
        if output_path is None:
            output_path = os.path.join(
                self._phos_viz_dir, 
                f"{unit.replace(':', '_')}_phos_vpm_{int(time.time())}.png"
            )
            
        save_img(img, output_path, title=f"SSP VPM (PHOS) - {unit}")
        
        return output_path

    # ---------------- Debugging ----------------
    
    def __repr__(self):
        """String representation for debugging."""
        episode_count = len(self._episode_traces)
        return (
            f"<VPMVisualizationService: status={'initialized' if self._initialized else 'uninitialized'}  "
            f"episodes={episode_count}  "
            f"dimensions={len(self._dimensions)}>"
        ) 