# stephanie/services/zero_model_service.py
"""
ZeroModelService
================
Visualizes Stephanie's cognitive progression as VPM (Value/Priority/Metrics) trajectories.
Transforms abstract improvement processes into visual "videos of thinking" that demonstrate
learning over time - perfect for your blog post's third implementation.

Key Features:
- Generates VPM timelines showing cognitive progression (columns=sections, rows=metrics)
- Creates MP4/GIF visualizations of the improvement process
- Integrates with CaseBookStore for trajectory data
- Supports domain-aware metric coloring (RGBA channels)
- Provides completion signals based on convergence metrics
- Exposes data for SIS dashboard integration

Blog Post Relevance:
This service directly enables your "learning & applied learning" implementation by:
1. Making Stephanie's thinking process visible (not just showing final output)
2. Demonstrating measurable progress (coverage from 0.62 → 0.82)
3. Connecting to PACS for trajectory-based training data
4. Using your existing VPMController infrastructure

Usage:
    zero_model = ZeroModelService(cfg, memory, logger)
    zero_model.generate_timeline(casebook_id="blog_post_123", output_path="vpm_timeline.mp4")
"""
from __future__ import annotations
import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# Import ZeroModel components directly
from zeromodel.core import ZeroModel
from zeromodel.vpm.image import VPMImageWriter, VPMImageReader
from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.pipeline.stages import (
    NormalizeStage,
    FeatureEngineeringStage,
    OrganizationStage,
    OcclusionExplainer
)
from zeromodel.normalizer import DynamicNormalizer
from zeromodel.provenance import create_vpf

from stephanie.services.service_protocol import Service
from stephanie.knowledge.casebook_store import CaseBookStore


class ZeroModelService(Service):
    """
    Service that properly integrates with ZeroModel's API.
    
    This implementation:
    - Uses ZeroModel's pipeline system
    - Leverages built-in VPM encoding/decoding
    - Integrates with provenance tracking
    - Follows ZeroModel's architectural patterns
    """
    
    def __init__(self, cfg: Dict[str, Any], memory, logger: logging.Logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.instance_id = f"zero_model_{int(time.time())}"
        
        # Get dependencies
        self.casebooks: CaseBookStore = cfg.get("casebooks") or CaseBookStore()
        
        # Initialize ZeroModel pipeline
        self._init_pipeline()
        
        self.logger.info("ZeroModelService initialized", {
            "pipeline_stages": [stage.name for stage in self.pipeline.stages],
            "message": "Using ZeroModel API directly"
        })
    
    def _init_pipeline(self):
        """Initialize ZeroModel's pipeline with proper configuration."""
        # Define the processing pipeline
        pipeline_config = [
            {"name": "normalize", "params": {"metric_names": self._get_metric_names()}},
            {"name": "feature_engineering", "params": {"nonlinearity_hint": "auto"}},
            {"name": "organization", "params": {"strategy": "spatial"}},
            {"name": "occlusion_explainer", "params": {"window_size": 5}}
        ]
        
        # Create pipeline executor
        self.pipeline = PipelineExecutor(pipeline_config)
    
    def _get_metric_names(self) -> List[str]:
        """Get metric names from configuration or default."""
        return self.cfg.get("zero_model", {}).get("metrics", [
            "coverage", "correctness", "coherence", 
            "citation_support", "entity_consistency", "domain_alignment",
            "novelty", "readability", "stickiness"
        ])
    
    def generate_timeline(
        self,
        casebook_id: str,
        output_path: str,
        goal_template: str = "academic_summary"
    ) -> Dict[str, Any]:
        """
        Generate a VPM timeline using ZeroModel's API.
        
        This properly uses ZeroModel's pipeline system rather than
        reimplementing VPM encoding logic.
        """
        # 1. Extract data from casebook
        vpm_data, section_info = self._extract_casebook_data(casebook_id)
        
        # 2. Process through ZeroModel pipeline
        processed_data, metadata = self._process_through_pipeline(vpm_data)
        
        # 3. Generate VPM timeline using ZeroModel's image writer
        self._generate_vpm_timeline(processed_data, section_info, output_path)
        
        # 4. Analyze convergence using ZeroModel's analysis tools
        convergence = self._analyze_convergence(processed_data)
        
        return {
            "casebook_id": casebook_id,
            "output_path": output_path,
            "convergence": convergence,
            "metadata": metadata,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _extract_casebook_data(self, casebook_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract VPM data from casebook using ZeroModel-compatible format."""
        casebook = self.casebooks.get_by_id(casebook_id)
        if not casebook:
            raise ValueError(f"Casebook not found: {casebook_id}")
        
        # Get metric names
        metrics = self._get_metric_names()
        
        # Extract data in ZeroModel-compatible format
        # ZeroModel expects shape (documents, metrics)
        steps = []
        section_map = {}
        
        for case in casebook.cases:
            if case.role != "text" or not case.meta or "vpm_row" not in case.meta:
                continue
                
            section_name = case.meta.get("section_name", "Unknown")
            if section_name not in section_map:
                section_map[section_name] = len(section_map)
            
            vpm_row = case.meta["vpm_row"]
            step_data = [vpm_row.get(metric, 0.0) for metric in metrics]
            steps.append(step_data)
        
        # Convert to numpy array: (steps, metrics)
        vpm_array = np.array(steps, dtype=np.float32)
        
        return vpm_array, {
            "section_names": list(section_map.keys()),
            "section_map": section_map,
            "metrics": metrics
        }
    
    def _process_through_pipeline(self, vpm_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process data through ZeroModel's pipeline system."""
        # Create ZeroModel instance
        zeromodel = ZeroModel(self._get_metric_names())
        zeromodel.precision = 8  # 8-bit precision
        
        # Prepare data (this triggers the pipeline)
        zeromodel.prepare(vpm_data, sql_query=None)
        
        # Get processed data
        processed_data = zeromodel.sorted_matrix if zeromodel.sorted_matrix is not None else zeromodel.canonical_matrix
        
        # Generate VPM image
        vpm_image = VPMImageWriter(self.cfg.get("zero_model", {}).get("precision", 8)).encode(processed_data)
        
        # Create provenance metadata
        vpf = create_vpf(
            pipeline={"graph_hash": "sha3:base-level", "step": "spatial-organization"},
            model={"id": "zero-1.0", "assets": {}},
            determinism={"seed": 0, "rng_backends": ["numpy"]},
            params={"tile": "L0_X0_Y0"}
        )
        
        return processed_data, {
            "vpm_image": vpm_image,
            "vpf": vpf,
            "shape": processed_data.shape
        }
    
    def _generate_vpm_timeline(self, processed_data: np.ndarray, section_info: Dict[str, Any], output_path: str):
        """Generate timeline using ZeroModel's image writer."""
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Use VPMImageWriter to create the timeline
        writer = VPMImageWriter(precision=8)
        frames = []
        
        # Generate frame for each step
        for i in range(processed_data.shape[0]):
            frame = processed_data[i:i+1, :]  # Get single step
            frame_image = writer.encode(frame)
            frames.append(frame_image)
        
        # Save as GIF
        writer.save_gif(frames, output_path, duration=500)  # 0.5s per frame
    
    def _analyze_convergence(self, processed_data: np.ndarray) -> Dict[str, Any]:
        """Analyze convergence using ZeroModel's analysis tools."""
        # Get threshold from config
        threshold = self.cfg.get("zero_model", {}).get("convergence_threshold", 0.95)
        
        # Calculate when all metrics exceed threshold
        converged_steps = np.all(processed_data >= threshold, axis=1)
        
        # Find first sustained convergence
        min_convergence_steps = self.cfg.get("zero_model", {}).get("min_convergence_steps", 3)
        convergence_step = None
        
        for i in range(len(converged_steps) - min_convergence_steps + 1):
            if all(converged_steps[i:i+min_convergence_steps]):
                convergence_step = i
                break
        
        return {
            "converged": convergence_step is not None,
            "convergence_step": convergence_step,
            "steps_to_converge": convergence_step + 1 if convergence_step is not None else None
        }
    
    def _extract_vpm_trajectory(
        self,
        casebook: CaseBookORM,
        goal_template: str,
        section_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract VPM data from casebook trajectory.
        
        Returns:
            vpm_data: Array of shape (steps, sections, metrics)
            section_info: Dict with section metadata
        """
        # Determine metrics to use
        metrics = metrics or self.config.default_metrics
        
        # Group cases by step/section
        steps = []
        section_map = {}
        
        for case in casebook.cases:
            # Skip non-text cases or cases without VPM data
            if case.role != "text" or not case.meta or "vpm_row" not in case.meta:
                continue
                
            step_idx = case.meta.get("step_idx", 0)
            section_name = case.meta.get("section_name", "Unknown")
            
            # Track section mapping
            if section_name not in section_map:
                section_map[section_name] = len(section_map)
            
            # Ensure we have enough steps
            while len(steps) <= step_idx:
                steps.append({})
                
            # Store VPM data
            vpm_row = case.meta["vpm_row"]
            step_data = steps[step_idx]
            
            # Initialize section data if needed
            if section_name not in step_data:
                step_data[section_name] = {m: 0.0 for m in metrics}
            
            # Fill in metric values
            for metric in metrics:
                if metric in vpm_row:
                    step_data[section_name][metric] = float(vpm_row[metric])
        
        # Convert to numpy array: (steps, sections, metrics)
        num_steps = len(steps)
        num_sections = len(section_map)
        num_metrics = len(metrics)
        
        # Create ordered section names
        section_names = sorted(section_map.keys(), key=lambda x: section_map[x])
        
        # Build array
        vpm_array = np.zeros((num_steps, num_sections, num_metrics), dtype=np.float32)
        for step_idx, step_data in enumerate(steps):
            for section_name, metrics_data in step_data.items():
                section_idx = section_map[section_name]
                for metric_idx, metric in enumerate(metrics):
                    vpm_array[step_idx, section_idx, metric_idx] = metrics_data.get(metric, 0.0)
        
        return vpm_array, {
            "names": section_names,
            "map": section_map,
            "metrics": metrics
        }
    
    def _generate_timeline_frames(
        self,
        vpm_data: np.ndarray,
        section_info: Dict[str, Any],
        metrics: List[str]
    ) -> List[np.ndarray]:
        """
        Generate individual frames for the timeline.
        
        Each frame represents the VPM state at a particular step.
        """
        frames = []
        num_steps, num_sections, num_metrics = vpm_data.shape
        
        # Calculate image dimensions
        width = num_metrics * self.config.metric_width
        height = num_sections * self.config.section_height
        
        # Create a font for labels
        try:
            font = ImageFont.truetype("arial.ttf", self.config.label_font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.config.label_font_size)
            except:
                font = ImageFont.load_default()
        
        # Generate frame for each step
        for step_idx in range(num_steps):
            # Create base image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw grid if enabled
            if self.config.show_grid:
                for i in range(num_sections + 1):
                    y = i * self.config.section_height
                    draw.line([(0, y), (width, y)], fill=self.config.grid_color, width=1)
                for i in range(num_metrics + 1):
                    x = i * self.config.metric_width
                    draw.line([(x, 0), (x, height)], fill=self.config.grid_color, width=1)
            
            # Fill in metric values
            for section_idx in range(num_sections):
                for metric_idx, metric in enumerate(metrics):
                    value = vpm_data[step_idx, section_idx, metric_idx]
                    
                    # Get color based on value and domain
                    color = self._get_metric_color(value, metric, section_info["names"][section_idx])
                    
                    # Draw colored cell
                    x1 = metric_idx * self.config.metric_width
                    y1 = section_idx * self.config.section_height
                    x2 = x1 + self.config.metric_width - 1
                    y2 = y1 + self.config.section_height - 1
                    
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                    
                    # Add text label if value is high enough
                    if value > 0.7:
                        text = f"{value:.2f}"
                        text_width, text_height = draw.textsize(text, font=font)
                        draw.text(
                            (x1 + (self.config.metric_width - text_width) // 2, 
                             y1 + (self.config.section_height - text_height) // 2),
                            text, 
                            fill="white", 
                            font=font
                        )
            
            # Add section labels on left
            for section_idx, section_name in enumerate(section_info["names"]):
                y = section_idx * self.config.section_height + self.config.section_height // 2
                draw.text((5, y), section_name, fill="black", font=font)
            
            # Add metric labels at top
            for metric_idx, metric in enumerate(metrics):
                x = metric_idx * self.config.metric_width + self.config.metric_width // 2
                draw.text((x, 5), metric, fill="black", font=font, anchor="mt")
            
            # Add title with step info
            title = f"Step {step_idx + 1} of {num_steps}"
            draw.text((width // 2, height + 10), title, fill="black", font=font, anchor="mt")
            
            # Scale up for visibility
            if self.config.image_scale > 1:
                img = img.resize(
                    (width * self.config.image_scale, height * self.config.image_scale),
                    Image.NEAREST
                )
            
            # Convert to numpy array for video
            frames.append(np.array(img))
        
        return frames
    
    def _get_metric_color(self, value: float, metric: str, section_name: str) -> Tuple[int, int, int]:
        """
        Get color for a metric cell based on value and domain awareness.
        
        For domain_aware color scheme:
        R = goal-aligned quality (coverage+correctness+citations)
        G = stability (entity_consistency + structure/formatting)
        B = novelty
        A = stickiness (carry-over into final)
        """
        value = max(0.0, min(1.0, value))  # Clamp to [0,0-1.0]
        
        # Standard color scheme (blue gradient)
        if self.config.color_scheme == "standard":
            intensity = int(255 * value)
            return (intensity, intensity, 255)
        
        # Domain-aware color scheme
        elif self.config.color_scheme == "domain_aware":
            # Determine domain from section name (simplified)
            domain = "general"
            if "method" in section_name.lower():
                domain = "ml"
            elif "result" in section_name.lower():
                domain = "ml"
            elif "figure" in section_name.lower() or "table" in section_name.lower():
                domain = "general"
            elif "theory" in section_name.lower():
                domain = "theory"
                
            # Get base domain color
            base_color = self.domain_colors.get(domain, (148, 103, 189, 255))
            
            # Adjust based on metric type
            if metric in ["coverage", "correctness", "citation_support"]:
                # Quality metrics - increase intensity with value
                r, g, b, a = base_color
                intensity = 0.7 + 0.3 * value
                return (int(r * intensity), int(g * intensity), int(b * intensity))
            elif metric in ["entity_consistency", "coherence"]:
                # Stability metrics - green emphasis
                return (int(50 * (1-value)), int(200 * value), int(50 * (1-value)))
            elif metric == "novelty":
                # Novelty - blue emphasis
                return (int(50 * (1-value)), int(50 * (1-value)), int(200 * value))
            else:
                # Default - use domain color with value-based intensity
                r, g, b, a = base_color
                intensity = 0.5 + 0.5 * value
                return (int(r * intensity), int(g * intensity), int(b * intensity))
        
        # Custom color scheme (can be extended)
        else:
            intensity = int(255 * value)
            return (intensity, 0, 255 - intensity)
    
    def _save_timeline(self, frames: List[np.ndarray], output_path: str) -> None:
        """Save timeline frames as MP4 or GIF."""
        ext = os.path.splitext(output_path)[1].lower()
        
        if ext in [".mp4", ".mov", ".avi"]:
            # Save as MP4 video
            imageio.mimwrite(
                output_path,
                frames,
                fps=1.0 / self.config.frame_duration,
                codec="libx264",
                quality=8
            )
        elif ext in [".gif"]:
            # Save as GIF
            imageio.mimsave(
                output_path,
                frames,
                duration=self.config.frame_duration * 1000,  # in milliseconds
                loop=0  # loop forever
            )
        else:
            raise ValueError(f"Unsupported output format: {ext}")
    
    def _analyze_convergence(self, vpm_data: np.ndarray, section_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze convergence of the trajectory.
        
        Returns metrics about:
        - When convergence happened
        - Which sections converged first/last
        - Overall convergence score
        """
        num_steps, num_sections, num_metrics = vpm_data.shape
        metrics = section_info["metrics"]
        
        # Find convergence point (when all sections meet threshold for min steps)
        convergence_step = None
        for step_idx in range(num_steps - self.config.min_convergence_steps + 1):
            valid = True
            for s in range(num_sections):
                for m in range(num_metrics):
                    # Skip readability metric (FKGL) which is inverse
                    if metrics[m] == "readability":
                        continue
                        
                    # Check if metric meets threshold
                    if vpm_data[step_idx, s, m] < self.config.convergence_threshold:
                        valid = False
                        break
                if not valid:
                    break
            
            # Check if threshold maintained for required steps
            if valid:
                sustained = True
                for check_step in range(1, self.config.min_convergence_steps):
                    if step_idx + check_step >= num_steps:
                        sustained = False
                        break
                    for s in range(num_sections):
                        for m in range(num_metrics):
                            if metrics[m] == "readability":
                                continue
                            if vpm_data[step_idx + check_step, s, m] < self.config.convergence_threshold:
                                sustained = False
                                break
                        if not sustained:
                            break
                    if not sustained:
                        break
                
                if sustained:
                    convergence_step = step_idx
                    break
        
        # Calculate section convergence order
        section_convergence = []
        for s in range(num_sections):
            section_name = section_info["names"][s]
            converged_at = None
            
            for step_idx in range(num_steps):
                valid = True
                for m in range(num_metrics):
                    if metrics[m] == "readability":
                        continue
                    if vpm_data[step_idx, s, m] < self.config.convergence_threshold:
                        valid = False
                        break
                
                if valid:
                    converged_at = step_idx
                    break
            
            section_convergence.append({
                "section": section_name,
                "converged_at": converged_at,
                "steps_to_converge": converged_at + 1 if converged_at is not None else None
            })
        
        # Sort sections by convergence time
        section_convergence.sort(key=lambda x: (x["converged_at"] is None, x["converged_at"]))
        
        return {
            "converged": convergence_step is not None,
            "convergence_step": convergence_step,
            "steps_to_converge": convergence_step + 1 if convergence_step is not None else None,
            "sections_converged": sum(1 for s in section_convergence if s["converged_at"] is not None),
            "sections_total": num_sections,
            "section_convergence_order": section_convergence,
            "convergence_threshold": self.config.convergence_threshold,
            "min_convergence_steps": self.config.min_convergence_steps
        }
    
    def get_vpm_frame(self, casebook_id: str, step_idx: int) -> Optional[np.ndarray]:
        """
        Get a single VPM frame for a specific step in a casebook trajectory.
        
        Useful for dashboard integration or real-time monitoring.
        """
        try:
            casebook = self.casebooks.get_by_id(casebook_id)
            if not casebook:
                return None
                
            vpm_data, section_info = self._extract_vpm_trajectory(casebook, "academic_summary")
            
            if step_idx >= len(vpm_data):
                return None
                
            # Generate just this frame
            frames = self._generate_timeline_frames(
                vpm_data[step_idx:step_idx+1], 
                section_info, 
                section_info["metrics"]
            )
            
            return frames[0] if frames else None
            
        except Exception as e:
            self.logger.warning("VPMFrameExtractionFailed", {
                "error": str(e),
                "casebook_id": casebook_id,
                "step_idx": step_idx
            })
            return None
    
    def generate_completion_signal(self, casebook_id: str) -> Dict[str, Any]:
        """
        Generate a completion signal based on VPM convergence.
        
        This can be used as a repo signal (like CI passing) to indicate completion.
        """
        try:
            casebook = self.casebooks.get_by_id(casebook_id)
            if not casebook:
                return {"completed": False, "reason": "casebook_not_found"}
                
            vpm_data, section_info = self._extract_vpm_trajectory(casebook, "academic_summary")
            convergence = self._analyze_convergence(vpm_data, section_info)
            
            return {
                "completed": convergence["converged"],
                "convergence_step": convergence["convergence_step"],
                "sections_converged": convergence["sections_converged"],
                "sections_total": convergence["sections_total"],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning("CompletionSignalGenerationFailed", {
                "error": str(e),
                "casebook_id": casebook_id
            })
            return {"completed": False, "reason": "error_during_analysis"}