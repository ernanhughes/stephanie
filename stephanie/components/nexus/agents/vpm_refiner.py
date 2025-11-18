"""
VPM (Visual Policy Map) Refiner Agent

This agent refines a single scorable into an optimized visual policy map through a series
of visual operations. It acts as a "visual surgeon + film director" that:

1. Takes the first scorable from context and computes its metrics
2. Generates an initial VPM using ZeroModel or fallback synthesis
3. Applies a state machine of visual operations (bootstrap, debias, bridge_tame, zoom_focus)
4. Maintains a filmstrip audit trail of the refinement process
5. Produces a final optimized VPM with structural improvements

The refinement process uses structural metrics (separability, bridge_proxy, etc.) to guide
visual operations toward better policy map organization and clarity.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.nexus.utils.visual_thought import (VisualThoughtOp,
                                                             VisualThoughtType)
from stephanie.components.nexus.vpm.maps import MapProvider
from stephanie.components.nexus.vpm.state_machine import (Thought,
                                                          ThoughtExecutor,
                                                          VPMGoal, VPMState,
                                                          compute_phi)
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_processor import ScorableProcessor
from stephanie.services.zeromodel_service import ZeroModelService
from stephanie.utils.vpm_utils import (detect_vpm_layout, ensure_chw_u8,
                                       vpm_quick_dump)

log = logging.getLogger(__name__)


@dataclass
class VPMRefinerConfig:
    """
    Configuration for VPM refinement process.
    
    Attributes:
        mode: Operation mode - "filmstrip" (with visual output) or "online" (minimal output)
        out_root: Root directory for output artifacts
        img_size: Base size for VPM images (width/height)
        min_vis_height: Minimum height for visual representation (expands 1px height images)
        max_steps: Maximum number of refinement steps
        utility_threshold: Early stopping threshold for state utility
        phi_threshold: Threshold for phi-based structural metrics
        occ_*: Occlusion parameters (reserved for future use)
        operation_costs: Cost weights for different visual operations used in BCS calculation
        operation_sequences: Declarative sequences for each refinement phase
        accept_policy: Policy for accepting operations ("delta", "bcs", or "either")
        accept_eps: Epsilon value for minimum improvement threshold
    """
    
    # Operation mode
    mode: str = "filmstrip"
    out_root: str = "runs/vpm_refiner"

    # Visualization parameters
    img_size: int = 256
    min_vis_height: int = 32

    # Control loop parameters
    max_steps: int = 6
    utility_threshold: float = 0.90
    phi_threshold: float = 0.90

    # Occlusion parameters (reserved for future use)
    occ_patch_h: int = 12
    occ_patch_w: int = 12
    occ_stride: int = 8
    occ_prior: str = "top_left"
    occ_channel_agg: str = "mean"

    # Operation costs for BCS calculation
    operation_costs: Dict[str, float] = field(default_factory=lambda: {
        "zoom": 1.0, "bbox": 0.3, "path": 0.4, "highlight": 0.5, "blur": 0.6, "logic": 0.2
    })

    # Declarative operation sequences for each refinement phase
    operation_sequences: Dict[str, List[Dict]] = field(default_factory=lambda: {
        "bootstrap": [
            {"type": "logic", "params": {"op": "NOT", "a": ("map", "uncertainty"), "dst": 1}},
            {"type": "logic", "params": {"op": "AND", "a": ("map", "quality"), "b": ("channel", 1), "dst": 0}},
            {"type": "logic", "params": {"op": "AND", "a": ("map", "novelty"), "b": ("channel", 1), "dst": 2}},
            {"type": "logic", "params": {"op": "OR",  "a": ("channel", 0), "b": ("channel", 2), "dst": 0}},
        ],
        "debias": [
            {"type": "logic", "params": {"op": "SUB", "a": ("channel", 0), "b": ("map", "risk"), "dst": 0, "blend": 0.35}}
        ],
        "bridge_tame": [
            {"type": "logic", "params": {"op": "SUB", "a": ("channel", 0), "b": ("map", "bridge"), "dst": 0, "blend": 0.50}}
        ],
    })

    # Acceptance policy parameters
    accept_policy: str = "either"
    accept_eps: float = 1e-6


class VPMRefinerAgent(BaseAgent):
    """
    VPM Refiner Agent - Optimizes Visual Policy Maps through visual operations.
    
    This agent takes a scorable, generates a Visual Policy Map (VPM), and refines it
    through a series of visual operations to improve structural metrics like separability
    and reduce bridge artifacts. The process is documented through a filmstrip showing
    the evolution of the VPM.
    
    Key phases:
    1. Bootstrap: Initial VPM construction and basic channel organization
    2. Debias: Risk and uncertainty reduction
    3. Bridge Tame: Bridge artifact suppression
    4. Zoom Focus: Dynamic zoom into interesting regions
    """
    
    name = "nexus_vpm_refiner"

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any):
        """
        Initialize the VPM Refiner Agent.
        
        Args:
            cfg: Configuration dictionary
            memory: Memory component for state management
            container: DI container for service access
            logger: Logger instance
        """
        super().__init__(cfg, memory, container, logger)
        
        # Initialize configuration with defaults
        self.cfg = VPMRefinerConfig(
            mode=str(cfg.get("mode", "filmstrip")),
            out_root=str(cfg.get("out_root", "runs/vpm_refiner")),
            img_size=int(cfg.get("img_size", 256)),
            max_steps=int(cfg.get("max_steps", 6)),
            utility_threshold=float(cfg.get("utility_threshold", 0.90)),
            min_vis_height=int(cfg.get("min_vis_height", 32)),
            phi_threshold=float(cfg.get("phi_threshold", 0.90)),
            occ_patch_h=int(cfg.get("occ_patch_h", 12)),
            occ_patch_w=int(cfg.get("occ_patch_w", 12)),
            occ_stride=int(cfg.get("occ_stride", 8)),
            occ_prior=str(cfg.get("occ_prior", "top_left")),
            occ_channel_agg=str(cfg.get("occ_channel_agg", "mean")),
        )

        # Initialize services
        self.zm: ZeroModelService = container.get("zeromodel")
        self.scorable_processor = ScorableProcessor(cfg=cfg, memory=memory, container=container, logger=logger)
        self.exec = ThoughtExecutor(visual_op_cost=self.cfg.operation_costs)
        self.map_provider = MapProvider(self.zm)
        
        log.debug("VPMRefinerAgent initialized with config: %s", self.cfg)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for VPM refinement.
        
        Process flow:
        1. Extract and validate scorables from context
        2. Generate initial VPM from scorable metrics
        3. Initialize state and compute structural metrics (phi)
        4. Run refinement phases (bootstrap, debias, bridge_tame, zoom_focus)
        5. Generate filmstrip artifacts and return results
        
        Args:
            context: Execution context containing scorables and pipeline state
            
        Returns:
            Updated context with refinement results and artifacts
        """
        log.info("Starting VPM refinement process")
        
        # Validate input
        scorables = list(context.get("scorables") or [])
        if not scorables:
            log.warning("No scorables supplied; nothing to refine.")
            context[self.output_key] = {"status": "no_scorables"}
            return context

        # Setup execution environment
        self.zm.initialize()
        seed = Scorable.from_dict(scorables[0])
        run_id = context.get("pipeline_run_id") or uuid.uuid4().hex[:8]
        run_dir = self._setup_run_directory(run_id)

        # Process scorable metrics
        metrics_values, metrics_columns = await self._extract_metrics(seed, context)
        
        # Generate VPM
        vpm_tensor, adapter_meta = await self._generate_vpm(seed, metrics_values, metrics_columns, run_dir)
        
        # Initialize refinement state
        state, initial_rgb = self._initialize_refinement_state(vpm_tensor, adapter_meta)
        
        # Execute refinement pipeline
        final_state, frames, steps_meta = await self._execute_refinement_pipeline(
            state, initial_rgb, run_dir
        )

        # Save artifacts and update context
        return self._finalize_execution(context, final_state, frames, steps_meta, run_id, run_dir)

    # --------------------------------------------------------------------------
    # Core Refinement Pipeline Methods
    # --------------------------------------------------------------------------

    async def _execute_refinement_pipeline(self, state: VPMState, initial_rgb: np.ndarray, 
                                         run_dir: Path) -> Tuple[VPMState, List[np.ndarray], List[Dict]]:
        """
        Execute the complete VPM refinement pipeline.
        
        Args:
            state: Initial VPM state
            initial_rgb: Original VPM in HWC format for comparison
            run_dir: Output directory for artifacts
            
        Returns:
            Tuple of (final_state, frames, steps_metadata)
        """
        frames: List[np.ndarray] = []
        steps_meta: List[Dict[str, Any]] = []

        # Frame 0 = initial state
        frames.append(self._compose_frame(initial_rgb, self._hwc(state.X), state.utility))

        # Phase 1: Bootstrap (once)
        state, bootstrap_rec = self._apply_sequence(state, "bootstrap")
        if bootstrap_rec:
            bootstrap_rec.update({"step": -1})
            steps_meta.append(bootstrap_rec)
            frames.append(self._compose_frame(initial_rgb, self._hwc(state.X), state.utility))

        # Phase 2: Iterative refinement
        for step in range(self.cfg.max_steps):
            if state.utility >= self.cfg.utility_threshold:
                log.info("Early stop at step %d: utility=%.4f ≥ threshold=%.4f", 
                           step, state.utility, self.cfg.utility_threshold)
                break

            # Apply debias and bridge_tame phases
            state = await self._apply_iteration_phases(state, step, initial_rgb, frames, steps_meta)

            # Apply zoom focus
            state = await self._apply_zoom_focus(state, step, initial_rgb, frames, steps_meta)

        return state, frames, steps_meta

    async def _apply_iteration_phases(self, state: VPMState, step: int, initial_rgb: np.ndarray,
                                    frames: List[np.ndarray], steps_meta: List[Dict]) -> VPMState:
        """Apply debias and bridge_tame phases for one iteration."""
        for phase in ("debias", "bridge_tame"):
            state, phase_rec = self._apply_sequence(state, phase)
            if phase_rec:
                phase_rec.update({"step": step})
                steps_meta.append(phase_rec)
                if phase_rec.get("accepted", True):
                    frames.append(self._compose_frame(initial_rgb, self._hwc(state.X), state.utility))
        return state

    async def _apply_zoom_focus(self, state: VPMState, step: int, initial_rgb: np.ndarray,
                              frames: List[np.ndarray], steps_meta: List[Dict]) -> VPMState:
        """Apply zoom focus operation."""
        cy, cx = self._attention_centroid(state.X[0], top_k_ratio=0.05)
        log.debug("Zoom focus target: center(x=%d, y=%d) scale=2.0", int(cx), int(cy))
        
        state, zoom_rec = self._apply_and_log(
            state,
            "zoom_focus",
            [VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (int(cx), int(cy)), "scale": 2.0})],
        )
        zoom_rec.update({"step": step})
        steps_meta.append(zoom_rec)
        if zoom_rec.get("accepted", True):
            frames.append(self._compose_frame(initial_rgb, self._hwc(state.X), state.utility))
            
        return state

    # --------------------------------------------------------------------------
    # Initialization Methods
    # --------------------------------------------------------------------------

    def _setup_run_directory(self, run_id: str) -> Path:
        """Setup and return the run directory for artifacts."""
        run_dir = Path(self.cfg.out_root) / f"{run_id}"
        (run_dir / "vpm_debug").mkdir(parents=True, exist_ok=True)
        log.info("VPMRefiner run_id=%s out_dir=%s", run_id, run_dir)
        return run_dir

    async def _extract_metrics(self, seed: Scorable, context: Dict[str, Any]) -> Tuple[List[float], List[str]]:
        """Extract metrics from scorable using ScorableProcessor."""
        row = await self.scorable_processor.process(seed, context=context)
        metrics_values = row.get("metrics_values", []) or []
        metrics_columns = row.get("metrics_columns", []) or []
        log.debug("Extracted metrics: %d columns, %d values", 
                    len(metrics_columns), len(metrics_values))
        return metrics_values, metrics_columns

    async def _generate_vpm(self, seed: Scorable, metrics_values: List[float], 
                          metrics_columns: List[str], run_dir: Path) -> Tuple[np.ndarray, Dict]:
        """Generate VPM tensor using ZeroModel or fallback method."""
        try:
            chw_u8, adapter_meta = await self.zm.vpm_from_scorable(
                seed, metrics_values=metrics_values, metrics_columns=metrics_columns
            )
            log.info("VPM generated via ZeroModel, layout: %s", detect_vpm_layout(chw_u8))
        except Exception as e:
            log.error("ZeroModel VPM generation failed: %s, using fallback", e)
            chw_u8 = self._fallback_vpm_from_metrics(metrics_values, metrics_columns)
            adapter_meta = {"fallback": True}

        # Debug dump and validation
        dump_info = vpm_quick_dump(chw_u8, run_dir / "vpm_debug", "sample")
        log.debug("VPM debug info: layout=%s shape=%s", dump_info["layout"], dump_info["shape"])
        
        return chw_u8, adapter_meta

    def _initialize_refinement_state(self, chw_u8: np.ndarray, 
                                   adapter_meta: Dict[str, Any]) -> Tuple[VPMState, np.ndarray]:
        """Initialize VPM state and compute initial structural metrics."""
        # Ensure proper visual format
        chw_u8 = self._ensure_visual_chw(chw_u8)
        C, H, W = chw_u8.shape
        log.debug("Visual tensor prepared: C=%d H=%d W=%d", C, H, W)

        # Store original for comparison
        initial_rgb = self._hwc(chw_u8)

        # Initialize state with structural metrics
        channel_semantics = adapter_meta.get("channel_semantics", {"node": 0, "edge": 1, "heat": 2})
        state = VPMState(
            X=chw_u8,
            meta={"adapter": adapter_meta, "channel_semantics": channel_semantics},
            phi=compute_phi(chw_u8, {"channel_semantics": channel_semantics}),
            goal=VPMGoal(weights={"separability": 1.0, "bridge_proxy": -0.5}),
        )

        # Initialize and augment maps
        state.meta["maps"] = self._initialize_maps(state.X)
        self._log_state("initial", state.phi, state.utility)

        return state, initial_rgb

    def _initialize_maps(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Initialize and augment maps for VPM refinement."""
        try:
            maps = self.map_provider.build(X).maps
            log.debug("MapProvider generated maps: %s", sorted(maps.keys()))
        except Exception as e:
            log.warning("MapProvider failed: %s, using fallback maps", e)
            maps = {}
            
        return self._augment_maps(maps, X)

    # --------------------------------------------------------------------------
    # Operation Application Methods
    # --------------------------------------------------------------------------

    def _apply_sequence(self, state: VPMState, sequence_name: str) -> Tuple[VPMState, Optional[Dict]]:
        """
        Apply a sequence of visual operations.
        
        Args:
            state: Current VPM state
            sequence_name: Name of the operation sequence to apply
            
        Returns:
            Tuple of (updated_state, execution_record)
        """
        sequence_specs = self.cfg.operation_sequences.get(sequence_name, [])
        executable_ops = self._build_executable_operations(sequence_specs, sequence_name, state)
        
        if not executable_ops:
            log.info("[%s] No executable operations", sequence_name)
            return state, None
            
        return self._apply_and_log(state, sequence_name, executable_ops)

    def _build_executable_operations(self, sequence_specs: List[Dict], 
                                   sequence_name: str, state: VPMState) -> List[VisualThoughtOp]:
        """Build executable operations from sequence specifications."""
        executable_ops = []
        
        for spec in sequence_specs:
            op_type = VisualThoughtType[spec["type"].upper()]
            params = dict(spec["params"])
            
            if not self._should_skip_operation(params, state.meta.get("maps", {})):
                executable_ops.append(VisualThoughtOp(op_type, params))
            else:
                log.debug("[%s] Skipping operation due to missing map dependencies", sequence_name)
                
        return executable_ops

    def _should_skip_operation(self, params: Dict, available_maps: Dict) -> bool:
        """Check if operation should be skipped due to missing map dependencies."""
        for key in ("a", "b"):
            source_spec = params.get(key)
            if (isinstance(source_spec, (tuple, list)) and 
                source_spec and source_spec[0] == "map" and 
                source_spec[1] not in available_maps):
                return True
        return False

    def _apply_and_log(self, state: VPMState, operation_name: str, 
                      operations: List[VisualThoughtOp]) -> Tuple[VPMState, Dict[str, Any]]:
        """
        Apply visual operations and log the results.
        
        Args:
            state: Current VPM state
            operation_name: Name of the operation being applied
            operations: List of visual operations to apply
            
        Returns:
            Tuple of (updated_state, execution_record)
        """
        # Log pre-application state
        phi_before = dict(state.phi)
        utility_before = state.utility
        self._log_state(f"{operation_name}:before", phi_before, utility_before)

        # Apply operations
        new_state, delta, cost, bcs = self.exec.score_thought(state, Thought(operation_name, operations))
        phi_after = dict(new_state.phi)
        utility_after = new_state.utility

        # Decision: accept or reject
        accepted = self._accept(delta, cost, bcs)
        
        if accepted:
            self._log_operation_accepted(operation_name, phi_before, phi_after, 
                                       utility_before, utility_after, cost, bcs)
            return new_state, self._create_operation_record(
                operation_name, True, phi_before, phi_after, 
                utility_before, utility_after, delta, cost, bcs
            )
        else:
            self._log_operation_rejected(operation_name, utility_before, utility_after, cost, bcs)
            return state, self._create_operation_record(
                operation_name, False, phi_before, phi_before,
                utility_before, utility_before, 0.0, cost, bcs
            )

    def _accept(self, delta: float, cost: float, bcs: float) -> bool:
        """
        Decide whether to accept an operation based on policy.
        
        Args:
            delta: Utility change
            cost: Operation cost
            bcs: Bridge Coherence Score
            
        Returns:
            True if operation should be accepted
        """
        policy = self.cfg.accept_policy
        eps = self.cfg.accept_eps
        
        if policy == "delta":
            return (delta is not None) and (float(delta) >= -eps)
        elif policy == "bcs":
            return (bcs is not None) and (float(bcs) >= -eps)
        else:  # "either"
            return ((delta is not None) and (float(delta) >= -eps)) or \
                   ((bcs is not None) and (float(bcs) >= -eps))

    # --------------------------------------------------------------------------
    # Visualization and Composition Methods
    # --------------------------------------------------------------------------

    def _compose_frame(self, top_rgb: np.ndarray, bottom_rgb: np.ndarray, 
                      utility: float) -> np.ndarray:
        """
        Compose a filmstrip frame showing before/after comparison with metrics.
        
        Args:
            top_rgb: Original VPM (HWC format)
            bottom_rgb: Current refined VPM (HWC format) 
            utility: Current utility value for metric bar
            
        Returns:
            Composite image as numpy array
        """
        assert top_rgb.shape == bottom_rgb.shape, "Top/bottom dimensions must match"
        
        H, W, _ = top_rgb.shape
        composite = np.vstack([top_rgb, bottom_rgb]).astype(np.uint8)
        
        # Add metric bar
        bar_height = max(18, int(0.12 * H))
        canvas = np.zeros((2 * H + bar_height, W, 3), dtype=np.uint8)
        canvas[:2 * H] = composite

        # Draw utility indicator
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Draw metric bar background
        draw.rectangle([(0, 2 * H), (W, 2 * H + bar_height)], fill=(0, 0, 0))
        
        # Draw utility percentage
        utility_percent = int(np.clip(utility, 0.0, 1.0) * 100.0 + 0.5)
        label = f"Utility: {utility_percent}%"
        
        # Calculate text position
        if hasattr(draw, "textbbox"):
            text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]
        else:
            text_width, text_height = (len(label) * 6, 12)
            
        x_pos = 10
        y_pos = 2 * H + (bar_height - text_height) // 2
        
        # Draw text with shadow for readability
        draw.text((x_pos + 1, y_pos + 1), label, fill=(0, 0, 0), font=font)
        draw.text((x_pos, y_pos), label, fill=(255, 255, 255), font=font)
        
        return np.asarray(img)

    def _attention_centroid(self, channel_data: np.ndarray, 
                          top_k_ratio: float = 0.05) -> Tuple[int, int]:
        """
        Calculate centroid of top-k attention mass.
        
        Args:
            channel_data: 2D array of channel data
            top_k_ratio: Ratio of top values to consider
            
        Returns:
            Tuple of (y, x) centroid coordinates
        """
        data = np.asarray(channel_data, dtype=np.float32)
        height, width = data.shape
        
        if data.max() <= 1e-9:
            return height // 2, width // 2
            
        flattened = data.ravel()
        k = max(1, int(round(top_k_ratio * flattened.size)))
        top_indices = np.argpartition(flattened, flattened.size - k)[-k:]
        
        y_coords, x_coords = np.divmod(top_indices, width)
        centroid_y = int(np.clip(np.mean(y_coords), 0, height - 1))
        centroid_x = int(np.clip(np.mean(x_coords), 0, width - 1))
        
        return centroid_y, centroid_x

    # --------------------------------------------------------------------------
    # Map Management Methods
    # --------------------------------------------------------------------------

    def _augment_maps(self, maps: Dict[str, np.ndarray], 
                     X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Augment maps with fallbacks for missing required maps.
        
        Ensures the presence of core maps (quality, novelty, uncertainty, risk, bridge)
        by creating fallback versions when they are missing.
        
        Args:
            maps: Original maps dictionary
            X: VPM tensor for fallback generation
            
        Returns:
            Augmented maps dictionary
        """
        height, width = X.shape[-2], X.shape[-1]
        augmented_maps = dict(maps or {})
        
        # Use channel 0 as attention proxy
        attention_proxy = self._normalize_to_01(X[0])

        # Ensure required maps exist with fallbacks
        if "quality" not in augmented_maps:
            augmented_maps["quality"] = attention_proxy
            
        if "novelty" not in augmented_maps:
            augmented_maps["novelty"] = np.clip(
                np.abs(attention_proxy - float(attention_proxy.mean())) * 2.0, 0.0, 1.0
            )
            
        if "uncertainty" not in augmented_maps:
            augmented_maps["uncertainty"] = 1.0 - attention_proxy
            
        if "risk" not in augmented_maps:
            if "uncertainty" in augmented_maps:
                augmented_maps["risk"] = augmented_maps["uncertainty"].copy()
                log.debug("Using uncertainty as risk fallback")
            else:
                quality = augmented_maps.get("quality", attention_proxy)
                augmented_maps["risk"] = np.clip(1.0 - quality, 0.0, 1.0)
                log.debug("Synthesized risk from inverse quality")
                
        if "bridge" not in augmented_maps:
            bridge_width = max(1, width // 16)
            center_mask = np.zeros((height, width), dtype=np.float32)
            left_margin = (width - bridge_width) // 2
            right_margin = left_margin + bridge_width
            center_mask[:, left_margin:right_margin] = 1.0
            augmented_maps["bridge"] = np.clip(attention_proxy * center_mask, 0.0, 1.0)

        return augmented_maps

    def _normalize_to_01(self, data: np.ndarray) -> np.ndarray:
        """Normalize array to 0-1 range."""
        normalized = data.astype(np.float32)
        if normalized.max() > 1.0:
            normalized = normalized / 255.0
        return np.clip(normalized, 0.0, 1.0)

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------

    def _fallback_vpm_from_metrics(self, values: List[float], 
                                 columns: List[str]) -> np.ndarray:
        """Generate fallback VPM from metrics when ZeroModel fails."""
        if not values:
            bar = np.zeros((1, 1, 64), dtype=np.uint8)
        else:
            normalized_values = np.asarray(values, dtype=np.float32)
            normalized_values = (normalized_values - np.nanmin(normalized_values)) / \
                              (np.nanmax(normalized_values) - np.nanmin(normalized_values) + 1e-9)
            bar = (normalized_values[None, None, :] * 255).astype(np.uint8)
            
        return np.repeat(bar, 3, axis=0)  # Convert to 3-channel

    def _ensure_visual_chw(self, tensor: np.ndarray) -> np.ndarray:
        """Ensure tensor is in proper visual CHW format."""
        visual_tensor = ensure_chw_u8(tensor, force_three=True)
        channels, height, width = visual_tensor.shape
        
        # Expand single-pixel height for visibility
        if height == 1 and self.cfg.min_vis_height > 1:
            repeats = max(1, int(np.ceil(self.cfg.min_vis_height / float(height))))
            visual_tensor = np.tile(visual_tensor, (1, repeats, 1))[:, :self.cfg.min_vis_height, :]
            
        return visual_tensor

    def _hwc(self, chw_tensor: np.ndarray) -> np.ndarray:
        """Convert CHW tensor to HWC format."""
        return np.transpose(chw_tensor, (1, 2, 0))

    # --------------------------------------------------------------------------
    # Logging Methods
    # --------------------------------------------------------------------------

    def _log_state(self, tag: str, phi: Dict[str, float], utility: float) -> None:
        """Log VPM state metrics."""
        log.info(
            "[%s] φ: sep=%.4f bridge=%.4f spec_gap=%.4f symmetry=%.4f crossings=%s | utility=%.4f",
            tag,
            float(phi.get("separability", 0.0)),
            float(phi.get("bridge_proxy", 0.0)),
            float(phi.get("spectral_gap", 0.0)),
            float(phi.get("vision_symmetry", 0.0)),
            str(phi.get("crossings", 0)),
            float(utility),
        )

    def _log_operation_accepted(self, op_name: str, phi_before: Dict, phi_after: Dict,
                              util_before: float, util_after: float, cost: float, bcs: float) -> None:
        """Log accepted operation details."""
        log.info(
            "[%s:accepted] Δφ: sep=%+0.4f bridge=%+0.4f spec_gap=%+0.4f symmetry=%+0.4f | "
            "Δutility=%+0.4f cost=%.3f bcs=%+.4f | utility=%.4f",
            op_name,
            float(phi_after.get("separability", 0.0)) - float(phi_before.get("separability", 0.0)),
            float(phi_after.get("bridge_proxy", 0.0)) - float(phi_before.get("bridge_proxy", 0.0)),
            float(phi_after.get("spectral_gap", 0.0)) - float(phi_before.get("spectral_gap", 0.0)),
            float(phi_after.get("vision_symmetry", 0.0)) - float(phi_before.get("vision_symmetry", 0.0)),
            float(util_after - util_before),
            float(cost),
            float(bcs),
            float(util_after),
        )

    def _log_operation_rejected(self, op_name: str, util_before: float, 
                              util_after: float, cost: float, bcs: float) -> None:
        """Log rejected operation details."""
        log.info(
            "[%s:rejected] Δutility=%.4f cost=%.3f bcs=%+.4f → REVERT", 
            op_name, float(util_after - util_before), float(cost), float(bcs)
        )

    def _create_operation_record(self, name: str, accepted: bool, phi_before: Dict,
                               phi_after: Dict, util_before: float, util_after: float,
                               delta: float, cost: float, bcs: float) -> Dict[str, Any]:
        """Create standardized operation record."""
        return {
            "name": name,
            "accepted": accepted,
            "phi_before": phi_before,
            "phi_after": phi_after,
            "utility_before": float(util_before),
            "utility_after": float(util_after),
            "delta_utility": float(delta),
            "cost": float(cost),
            "bcs": float(bcs),
        }

    def _finalize_execution(self, context: Dict[str, Any], final_state: VPMState,
                          frames: List[np.ndarray], steps_meta: List[Dict], 
                          run_id: str, run_dir: Path) -> Dict[str, Any]:
        """Finalize execution by saving artifacts and updating context."""
        # Save filmstrip if in filmstrip mode
        if self.cfg.mode == "filmstrip" and frames:
            self._save_filmstrip_artifacts(run_dir, frames, steps_meta)
            context["vpm_artifacts"] = {"run_dir": str(run_dir), "frames": len(frames)}

        # Update context with results
        context[self.output_key] = {
            "status": "ok",
            "final_utility": float(final_state.utility),
            "steps": steps_meta,
            "run_id": run_id,
            "artifacts_dir": str(run_dir),
        }

        log.info("VPM refinement completed: final_utility=%.4f, steps=%d, artifacts=%s",
                   final_state.utility, len(steps_meta), run_dir)

        return context

    def _save_filmstrip_artifacts(self, run_dir: Path, frames: List[np.ndarray], 
                                steps_meta: List[Dict]) -> None:
        """Save filmstrip artifacts (PNGs, GIF, metadata)."""
        # Save individual frames
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(run_dir / f"frame_{i:02d}.png")

        # Save animated GIF
        iio.mimsave(run_dir / "filmstrip.gif", frames, fps=2, loop=0)

        # Save metrics metadata
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"steps": steps_meta}, f, indent=2)

        log.info("Filmstrip artifacts saved: %d frames → %s", len(frames), run_dir)