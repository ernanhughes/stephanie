from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

# Initialize the simple logging mechanism for the module
log = logging.getLogger(__name__)


# ---------------------------- Config -----------------------------------------

@dataclass
class VPMRefinerConfig:
    """
    Configuration for the VPMRefinerAgent. Defines the refinement objective,
    loop control, visualization mode, and operational costs.
    """
    mode: str = field(
        default="filmstrip",
        metadata={"help": "'online' (minimal logging) or 'filmstrip' (save frames/GIF/JSON)."}
    )
    out_root: str = field(
        default="runs/vpm_refiner",
        metadata={"help": "Root directory for saving run artifacts (frames, GIF, metrics.json)."}
    )

    # Visualization and input standardization
    img_size: int = field(
        default=256,
        metadata={"help": "Base VPM size (width/height) expected from ZeroModel or used for fallback."}
    )
    min_vis_height: int = field(
        default=32,
        metadata={"help": "Minimum height for the visual representation. 1xW input VPMs are tiled vertically to this size for readability."}
    )

    # Control loop
    max_steps: int = field(
        default=6,
        metadata={"help": "Total budget for refinement steps (debias/bridge_tame/zoom cycles)."}
    )
    utility_threshold: float = field(
        default=0.90,
        metadata={"help": "Stop refinement if VPMState.utility reaches or exceeds this value."}
    )
    phi_threshold: float = field(
        default=0.90,
        metadata={"help": "Threshold for structural metrics (phi); primarily for logging/future use."}
    )

    # Occlusion parameters (reserved for future use / model interpretation)
    occ_patch_h: int = field(default=12, metadata={"help": "Occlusion patch height (reserved)."})
    occ_patch_w: int = field(default=12, metadata={"help": "Occlusion patch width (reserved)."})
    occ_stride: int = field(default=8, metadata={"help": "Occlusion stride (reserved)."})
    occ_prior: str = field(default="top_left", metadata={"help": "Occlusion prior strategy (reserved)."})
    occ_channel_agg: str = field(default="mean", metadata={"help": "Occlusion channel aggregation (reserved)."})

    # Operation costs
    operation_costs: Dict[str, float] = field(default_factory=lambda: {
        "zoom": 1.0, "bbox": 0.3, "path": 0.4, "highlight": 0.5, "blur": 0.6, "logic": 0.2
    }, metadata={"help": "Per-operation penalty used by ThoughtExecutor to calculate BCS (utility - cost)."})

    # Declarative operation sequences for refinement phases
    operation_sequences: Dict[str, List[Dict]] = field(default_factory=lambda: {
        "bootstrap": [
            # 1. Calculate an inverted uncertainty map (NOT uncertainty) and store in channel 1.
            {"type": "logic", "params": {"op": "NOT", "a": ("map", "uncertainty"), "dst": 1}},
            # 2. Channel 0 (primary attention): Quality AND (NOT Uncertainty)
            {"type": "logic", "params": {"op": "AND", "a": ("map", "quality"), "b": ("channel", 1), "dst": 0}},
            # 3. Channel 2 (auxiliary): Novelty AND (NOT Uncertainty)
            {"type": "logic", "params": {"op": "AND", "a": ("map", "novelty"), "b": ("channel", 1), "dst": 2}},
            # 4. Final Channel 0: Quality/Novelty-gated signals (main goodness signal).
            {"type": "logic", "params": {"op": "OR",  "a": ("channel", 0), "b": ("channel", 2), "dst": 0}},
        ],
        "debias": [
            # Soft subtraction of 'risk' from channel 0. Blend ensures a gradual, non-zeroing change.
            {"type": "logic", "params": {"op": "SUB", "a": ("channel", 0), "b": ("map", "risk"), "dst": 0, "blend": 0.35}}
        ],
        "bridge_tame": [
            # Soft subtraction of 'bridge' activation from channel 0 to penalize over-connectivity.
            {"type": "logic", "params": {"op": "SUB", "a": ("channel", 0), "b": ("map", "bridge"), "dst": 0, "blend": 0.50}}
        ],
    }, metadata={"help": "Declarative sequence of VisualThoughtOps for named refinement phases (bootstrap, debias, bridge_tame)."})

    # Acceptance policy
    accept_policy: str = field(
        default="either",
        metadata={"help": "Which metric governs accept/reject: 'delta' (Δutility>=0), 'bcs' (Δutility - cost >= 0), or 'either' (pass if delta OR bcs is non-negative)."}
    )
    accept_eps: float = field(
        default=1e-6,
        metadata={"help": "Minimum improvement threshold (epsilon) for accepting a step."}
    )


# ---------------------------- Agent ------------------------------------------

class VPMRefinerAgent(BaseAgent):
    """
    The VPMRefinerAgent is a visual thought engine that refines a Visual Policy Map (VPM)
    for a single Scorable under a structural 'VPMGoal'.

    Its core function is:
    1. Metrics extraction for the Scorable.
    2. VPM generation (via ZeroModel or fallback).
    3. Generation of interpretation maps (risk, quality, novelty, etc.).
    4. Running a small, scripted refinement program (bootstrap, debias, bridge_tame, zoom_focus).
    5. Logging every accepted step as a frame in an optional 'filmstrip' audit trail.

    It aims to push the VPM toward a structure with high separability and low bridging.
    """
    name = "nexus_vpm_refiner"
    cfg: VPMRefinerConfig # Type hint for config

    def __init__(self, cfg, memory, container, logger):
        """Initializes the agent and its core components."""
        super().__init__(cfg, memory, container, logger)
        # Apply raw config dict to the validated VPMRefinerConfig dataclass
        self.cfg = VPMRefinerConfig(
            **{k: v for k, v in cfg.items() if hasattr(VPMRefinerConfig, k)}
        )

        # Dependency Injection: wire up external services
        self.zm: ZeroModelService = container.get("zeromodel")
        self.scorable_processor = ScorableProcessor(
            cfg=cfg, memory=memory, container=container, logger=logger
        )
        # ThoughtExecutor calculates utility, cost, delta, and BCS for visual operations
        self.exec = ThoughtExecutor(visual_op_cost=self.cfg.operation_costs)
        # MapProvider generates channels like 'risk' or 'novelty' over the VPM
        self.map_provider = MapProvider(self.zm)
        log.debug("VPMRefinerAgent initialized with mode='%s'", self.cfg.mode)


    # ------------------------------------------------------------------ run --

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution path for the agent.

        Args:
            context: Dictionary containing input data. Must include 'scorables'
                     (list of dicts) and optionally 'pipeline_run_id'.

        Returns:
            The updated context dictionary with the refinement result under self.output_key.
        """
        scorables = list(context.get("scorables") or [])
        if not scorables:
            log.warning("No scorables supplied; nothing to refine.")
            context[self.output_key] = {"status": "no_scorables"}
            return context

        self.zm.initialize()
        seed = Scorable.from_dict(scorables[0])

        run_id = context.get("pipeline_run_id") or uuid.uuid4().hex[:8]
        run_dir = Path(self.cfg.out_root) / f"{run_id}"
        (run_dir / "vpm_debug").mkdir(parents=True, exist_ok=True)
        log.info("=== VPMRefiner run_id=%s out_dir=%s ===", run_id, run_dir)

        # 1) Metrics Extraction
        row = await self.scorable_processor.process(seed, context=context)
        metrics_values = row.get("metrics_values", []) or []
        metrics_columns = row.get("metrics_columns", []) or []
        log.debug("[metrics] columns=%d values=%d", len(metrics_columns), len(metrics_values))

        # 2) VPM Generation (ZeroModel or Fallback)
        try:
            chw_u8, adapter_meta = await self.zm.vpm_from_scorable(
                seed, metrics_values=metrics_values, metrics_columns=metrics_columns
            )
        except Exception as e:
            log.error("vpm_from_scorable failed: %s. Using fallback VPM.", e, exc_info=True)
            chw_u8, adapter_meta = self._fallback_vpm_from_metrics(metrics_values, metrics_columns), {"fallback": True}
        
        log.info("VPM layout from adapter: %s shape=%s", detect_vpm_layout(chw_u8), getattr(chw_u8, "shape", None))
        
        dump = vpm_quick_dump(chw_u8, run_dir / "vpm_debug", "sample")
        log.debug("VPM quick dump: layout=%s shape=%s gray=%s", dump["layout"], dump["shape"], dump["gray_path"])

        # 3) VPM Standardization and Initial Frame Setup
        # Ensures VPM is (C,H,W) uint8, has 3 channels, and meets minimum visual height.
        chw_u8 = self._ensure_visual_chw(chw_u8)
        C, H, W = chw_u8.shape
        log.debug("Post-visual ensure: C=%d H=%d W=%d", C, H, W)

        # HWC (Height, Width, Channel) format for PIL/ImageIO writing
        initial_rgb = self._hwc(chw_u8)

        # 4) Initial State, Goal, and Maps
        channel_semantics = adapter_meta.get("channel_semantics", {"node": 0, "edge": 1, "heat": 2})
        
        # Build initial state: X=VPM, phi=initial structural metrics, goal=utility objective.
        initial_phi = compute_phi(chw_u8, {"channel_semantics": channel_semantics})
        state = VPMState(
            X=chw_u8,
            meta={"adapter": adapter_meta, "channel_semantics": channel_semantics, "maps": {}},
            phi=initial_phi,
            # Goal: Maximize separability (1.0), Minimize bridge-proxy (-0.5)
            goal=VPMGoal(weights={"separability": 1.0, "bridge_proxy": -0.5}),
        )
        self._log_state("initial", state.phi, state.utility)

        # Build Interpretation Maps (risk, quality, etc.)
        try:
            maps = self.map_provider.build(state.X).maps
            log.debug("MapProvider maps: %s", sorted(list(maps.keys())))
        except Exception as e:
            log.warning("MapProvider.build failed: %s; using fallbacks.", e)
            maps = {}
        # Ensure canonical map keys exist (even if empty) and normalize them.
        state.meta["maps"] = self._augment_maps(maps, state.X)

        frames: List[np.ndarray] = []
        steps_meta: List[Dict[str, Any]] = []

        # Frame 0 is the initial VPM before any refinement
        frames.append(self._compose_frame(initial_rgb, self._hwc(state.X), state.utility, state.phi, 0))

        # 5) Refinement Phases
        state, steps_meta, frames = self._phase_bootstrap(state, steps_meta, frames, initial_rgb)
        
        state, steps_meta, frames = self._phase_debias_and_tame_loop(state, steps_meta, frames, initial_rgb)
        
        state, steps_meta, frames = self._phase_zoom_focus(state, steps_meta, frames, initial_rgb)


        # 6) Artifact Writing / Filmstrip Persistency
        if self.cfg.mode == "filmstrip":
            self._save_filmstrip(run_dir, frames, steps_meta)
            context["vpm_artifacts"] = {"run_dir": str(run_dir), "frames": len(frames)}

        context[self.output_key] = {
            "status": "ok",
            "final_utility": float(state.utility),
            "steps": steps_meta,
            "run_id": run_id,
            "dir": str(run_dir),
            "phi_final": state.phi,
        }
        log.info("final utility: %.4f", state.utility)
        log.info("artifacts dir: %s", run_dir)
        return context

    # ------------------------------------------------------------ Phases ----
    
    def _phase_bootstrap(self, state: VPMState, steps_meta: List[Dict], frames: List[np.ndarray], initial_rgb: np.ndarray) -> Tuple[VPMState, List[Dict], List[np.ndarray]]:
        """Applies the initial 'bootstrap' logic sequence once to set up primary channels."""
        log.info("Starting refinement phase: bootstrap (step 0)")
        
        # Use _apply_sequence to execute the ops defined in config under "bootstrap"
        new_state, rec = self._apply_sequence(state, "bootstrap")
        
        if rec and rec.get("accepted", True):
            rec.update({"step": 0, "phase": "bootstrap"})
            steps_meta.append(rec)
            frames.append(self._compose_frame(initial_rgb, self._hwc(new_state.X), new_state.utility, new_state.phi, len(steps_meta)))
            state = new_state
        elif rec:
            log.warning("Bootstrap phase rejected, continuing with initial state.")
            rec.update({"step": 0, "phase": "bootstrap"})
            steps_meta.append(rec) # Still log the rejected step
            
        return state, steps_meta, frames

    def _phase_debias_and_tame_loop(self, state: VPMState, steps_meta: List[Dict], frames: List[np.ndarray], initial_rgb: np.ndarray) -> Tuple[VPMState, List[Dict], List[np.ndarray]]:
        """Iteratively applies the debias and bridge_tame sequences."""
        for step in range(1, self.cfg.max_steps + 1):
            if state.utility >= self.cfg.utility_threshold:
                log.info("early-stop: step=%d utility=%.4f ≥ %.4f", step, state.utility, self.cfg.utility_threshold)
                break

            for phase_name in ("debias", "bridge_tame"):
                log.info("Starting refinement phase: %s (step %d)", phase_name, step)
                new_state, rec = self._apply_sequence(state, phase_name)
                
                if rec:
                    rec.update({"step": step, "phase": phase_name})
                    steps_meta.append(rec)
                    
                    if rec.get("accepted", True):
                        frames.append(self._compose_frame(initial_rgb, self._hwc(new_state.X), new_state.utility, new_state.phi, len(steps_meta)))
                        state = new_state
                    else:
                        log.debug("Phase '%s' rejected, state remains unchanged.", phase_name)

        return state, steps_meta, frames

    def _phase_zoom_focus(self, state: VPMState, steps_meta: List[Dict], frames: List[np.ndarray], initial_rgb: np.ndarray) -> Tuple[VPMState, List[Dict], List[np.ndarray]]:
        """Dynamically builds and applies a ZOOM operation based on attention centroid."""
        step = len(steps_meta) + 1
        log.info("Starting refinement phase: zoom_focus (step %d)", step)
        
        # Determine the region of interest for zooming. Uses channel 0 (main signal)
        # with a small top-k ratio to focus on the most important cluster mass.
        cy, cx = self._attention_centroid(state.X[0], top_k_ratio=0.05)
        log.debug("Zoom focus target: center(x=%d,y=%d) scale=2.0", int(cx), int(cy))
        
        zoom_op = VisualThoughtOp(
            VisualThoughtType.ZOOM, 
            {"center": (int(cx), int(cy)), "scale": 2.0}
        )
        
        # Apply the dynamic zoom operation
        new_state, rec = self._apply_and_log(
            state,
            "zoom_focus",
            [zoom_op],
        )
        
        rec.update({"step": step, "phase": "zoom_focus"})
        steps_meta.append(rec)
        
        if rec.get("accepted", True):
            frames.append(self._compose_frame(initial_rgb, self._hwc(new_state.X), new_state.utility, new_state.phi, len(steps_meta)))
            state = new_state
            
        return state, steps_meta, frames

    # ------------------------------------------------------------ Refinement Logic ----

    def _apply_and_log(self, state: VPMState, name: str, ops: List[VisualThoughtOp]) -> Tuple[VPMState, Dict[str, Any]]:
        """Apply a thought (list of ops), calculate metrics, and log the result. Implements the greedy accept policy."""
        before = dict(state.phi)
        u0 = state.utility
        self._log_state(f"{name}:before", before, u0)

        # Score the thought: returns new state, utility change (delta), cost, and BCS
        new_state, delta, cost, bcs = self.exec.score_thought(state, Thought(name, ops))
        after = dict(new_state.phi)
        u1 = new_state.utility

        accepted = self._accept(delta, cost, bcs)
        
        # Decision semantics
        if not accepted:
            # Reject: Return the original state and log rejection
            log.info("[%s:rejected] Δutility=%.4f cost=%.3f bcs=%+.4f → REVERT", name, float(u1 - u0), float(cost), float(bcs))
            self._log_state(f"{name}:after(REVERTED)", before, u0)
            return state, {
                "name": name, "accepted": False, "phi_before": before, "phi_after": before, 
                "utility_before": float(u0), "utility_after": float(u0), "delta_utility": 0.0,
                "cost": float(cost), "bcs": float(bcs), "ops": [op.to_dict() for op in ops]
            }

        # Accept: Log the positive change and return the new state
        log.info(
            "[%s:accepted] Δφ: sep=%+0.4f bridge=%+0.4f | Δutility=%+0.4f cost=%.3f bcs=%+.4f | utility=%.4f",
            name,
            float(after.get("separability", 0.0)) - float(before.get("separability", 0.0)),
            float(after.get("bridge_proxy", 0.0)) - float(before.get("bridge_proxy", 0.0)),
            float(u1 - u0), float(cost), float(bcs), float(u1),
        )
        return new_state, {
            "name": name, "accepted": True, "phi_before": before, "phi_after": after, 
            "utility_before": float(u0), "utility_after": float(u1), "delta_utility": float(u1 - u0),
            "cost": float(cost), "bcs": float(bcs), "ops": [op.to_dict() for op in ops]
        }

    def _accept(self, delta: float, cost: float, bcs: float) -> bool:
        """
        Decides whether to accept a proposed thought based on the configured policy.

        Acceptance conditions:
        - 'delta': (Δutility) >= -epsilon
        - 'bcs': (Δutility - cost) >= -epsilon
        - 'either': (Δutility OR BCS) >= -epsilon
        """
        eps = self.cfg.accept_eps
        
        if self.cfg.accept_policy == "delta":
            return (delta is not None) and (float(delta) >= -eps)
        if self.cfg.accept_policy == "bcs":
            return (bcs is not None) and (float(bcs) >= -eps)
        
        # Default is "either"
        return ((delta is not None) and (float(delta) >= -eps)) or \
               ((bcs is not None) and (float(bcs) >= -eps))

    def _apply_sequence(self, state: VPMState, name: str) -> Tuple[VPMState, Dict[str, Any] | None]:
        """
        Translates a declarative sequence of operations from config into VisualThoughtOps,
        applies them as a single Thought, and logs the result.
        """
        seq = self.cfg.operation_sequences.get(name, [])
        ops: List[VisualThoughtOp] = []
        
        # 1. Translate config dicts to VisualThoughtOp objects
        for spec in seq:
            typ = VisualThoughtType[spec["type"].upper()]
            params = dict(spec["params"])
            
            # Skip op if a required ("map", k) source is missing from the state
            skip = False
            for key in ("a", "b"):
                src = params.get(key)
                if isinstance(src, (tuple, list)) and src and src[0] == "map":
                    if src[1] not in state.meta.get("maps", {}):
                        log.debug("[%s] skip op due to missing map: %s", name, src[1])
                        skip = True
                        break
            if not skip:
                ops.append(VisualThoughtOp(typ, params))
        
        if not ops:
            log.debug("Sequence '%s' resulted in no executable operations.", name)
            return state, None
            
        # 2. Apply the compiled list of operations
        return self._apply_and_log(state, name, ops)

    # ------------------------------------------------------------ Helpers ----

    def _log_state(self, tag: str, phi: Dict[str, float], util: float) -> None:
        """Standardized logger for tracking VPM structural metrics (phi) and utility."""
        log.info(
            "[%s] φ: sep=%.4f bridge=%.4f spec_gap=%.4f | utility=%.4f",
            tag,
            float(phi.get("separability", 0.0)),
            float(phi.get("bridge_proxy", 0.0)),
            float(phi.get("spectral_gap", 0.0)),
            float(util),
        )
        log.debug(
            "[%s] full φ: symmetry=%.4f crossings=%s",
            tag,
            float(phi.get("vision_symmetry", 0.0)),
            str(phi.get("crossings", 0)),
        )

    def _fallback_vpm_from_metrics(self, values: List[float], columns: List[str]) -> np.ndarray:
        """Generates a simple VPM bar from raw metrics if ZeroModel fails."""
        log.info("Generating fallback VPM from %d metrics.", len(values))
        if not values:
            # Fallback to a small black bar if no metrics are present
            bar = np.zeros((1, 1, self.cfg.img_size), dtype=np.uint8)
        else:
            # Normalize metrics (0-1), tile to a 1x1xW bar, and scale to 0-255
            v = np.asarray(values, dtype=np.float32)
            v = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9)
            # Clip to a fixed size if needed, but here it's [1,1,len(values)]
            bar = (v[None, None, :] * 255).astype(np.uint8)
        
        # Ensure CHW format, repeat the single row to match target size, and tile to 3 channels (RGB)
        bar = np.repeat(bar, 3, axis=0)  # CHW (tile to 3ch)
        bar = self._ensure_visual_chw(bar)
        
        return bar

    def _ensure_visual_chw(self, chw: np.ndarray) -> np.ndarray:
        """Standardizes VPM to (C,H,W) uint8, enforces 3 channels, and ensures minimum height."""
        # Standardize to (C,H,W) uint8 and ensure 3 channels
        X = ensure_chw_u8(chw, force_three=True)
        C, H, W = X.shape
        
        # Enforce minimum height for visualization purposes (e.g., 1xW metrics bar)
        if H < self.cfg.min_vis_height:
            reps = max(1, int(np.ceil(self.cfg.min_vis_height / float(H))))
            # Tile rows vertically, then slice to exact min_vis_height
            X = np.tile(X, (1, reps, 1))[:, : self.cfg.min_vis_height, :]
            log.debug("Resized VPM from H=%d to H=%d for visual minimum.", H, X.shape[1])
            
        return X

    def _augment_maps(self, maps: Dict[str, np.ndarray], X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Ensures a comprehensive map dictionary by adding fallbacks for standard keys
        and normalizing all map tensors to 0-255 uint8.
        """
        C, H, W = X.shape
        # Ensure all canonical map keys exist (even if only as a zero-map fallback)
        canonical_keys = {"risk", "bridge", "quality", "novelty", "uncertainty"}
        
        for key in canonical_keys:
            if key not in maps:
                # Add a zero map of shape (H, W) as a safe fallback
                maps[key] = np.zeros((H, W), dtype=np.float32)
                log.debug("Added zero-map fallback for missing key: %s", key)
        
        # Normalize all map tensors to 0-255 uint8 for consistent consumption by ThoughtExecutor
        for k, v in maps.items():
            if v.dtype != np.uint8:
                # Normalize float maps to 0-255
                v = v.astype(np.float32)
                v_min, v_max = np.nanmin(v), np.nanmax(v)
                if v_max > v_min + 1e-6:
                    v = (v - v_min) / (v_max - v_min)
                maps[k] = (v * 255).astype(np.uint8)
            else:
                maps[k] = v # Already uint8
        
        return maps

    def _attention_centroid(self, attention_map: np.ndarray, top_k_ratio: float = 0.05) -> Tuple[float, float]:
        """
        Computes the weighted center of mass (centroid) of the top-k mass in the attention map.
        Used to determine the focus point for the ZOOM operation.
        Returns (cy, cx).
        """
        if attention_map.ndim == 3: # Expects HxW map, if C H W is passed, use first channel
            attention_map = attention_map[0]
        
        H, W = attention_map.shape
        
        # Flatten and find threshold for top_k_ratio mass
        flat_map = attention_map.flatten()
        threshold_idx = int(len(flat_map) * (1.0 - top_k_ratio))
        
        if threshold_idx >= len(flat_map):
            # Fallback: use center if map is empty or too small
            return H // 2, W // 2
        
        flat_map = np.sort(flat_map)
        threshold = flat_map[threshold_idx]

        # Only consider mass above the threshold
        mask = attention_map >= threshold
        
        # Compute indices of the mask
        y_indices, x_indices = np.where(mask)
        
        if not y_indices.size:
            # Second fallback: just use the geometric center
            return H // 2, W // 2
            
        # Get corresponding weights for the top-k mass
        weights = attention_map[mask]
        total_weight = np.sum(weights)

        if total_weight == 0:
             # Final fallback: use the geometric center if weights sum to zero
            return H // 2, W // 2
        
        # Weighted center of mass calculation (centroid)
        cy = np.sum(y_indices * weights) / total_weight
        cx = np.sum(x_indices * weights) / total_weight
        
        return float(cy), float(cx)

    # ------------------------------------------------------------ Visualization ----
    
    def _hwc(self, chw: np.ndarray) -> np.ndarray:
        """Converts (C, H, W) VPM tensor to (H, W, C) for image output."""
        return np.transpose(chw, (1, 2, 0))

    def _metric_bar(self, phi: Dict[str, float], utility: float, step_idx: int) -> np.ndarray:
        """
        Constructs a small HWC bar visualizing key metrics as color-coded stripes.
        This provides a quantitative audit trail embedded in the visual frame.
        """
        H, W = 32, 256
        bar = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Normalize structural metrics to a color scale (0-1.0)
        metrics = {
            "SEP": phi.get("separability", 0.0), # Reward (Green/Good)
            "BRIDGE": phi.get("bridge_proxy", 0.0), # Penalty (Red/Bad)
            "SPEC": phi.get("spectral_gap", 0.0),
        }
        
        # Calculate color for each metric and draw stripes
        x_start = 0
        for name, value in metrics.items():
            value = np.clip(value, 0.0, 1.0) # Ensure 0-1 range
            
            # Color coding: Green for good (SEP), Red for bad (BRIDGE)
            if name == "SEP":
                color = (int(value * 255), int((1 - value) * 128), 0)
            elif name == "BRIDGE":
                color = (int(value * 255), int((1 - value) * 128), 0) 
            else: # Grey scale for neutral metrics
                gray = int(value * 255)
                color = (gray, gray, gray)
                
            stripe_width = W // len(metrics) // 2
            x_end = x_start + stripe_width
            bar[:, x_start:x_end, :] = color # Color stripe
            
            # Label the stripe
            mid_x = x_start + stripe_width // 2
            # Add text label (requires PIL to draw text on NumPy array)
            
            x_start = x_end + stripe_width # Move to next position

        # Text Overlay: Utility and Step Index
        img_pil = Image.fromarray(bar)
        draw = ImageDraw.Draw(img_pil)
        
        # Use a common font or a fallback
        try:
            font = ImageFont.truetype("Arial.ttf", 10)
        except IOError:
            font = ImageFont.load_default() 
            
        text = f"STEP: {step_idx} | UTIL: {utility:.4f}"
        
        # Position text at the center-right
        text_w, text_h = draw.textsize(text, font=font)
        text_x = W - text_w - 5
        text_y = (H - text_h) // 2
        
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        
        return np.asarray(img_pil)

    def _compose_frame(self, initial_rgb: np.ndarray, current_rgb: np.ndarray, utility: float, phi: Dict[str, float], step_idx: int) -> np.ndarray:
        """
        Composes the final visualization frame: [Initial VPM] / [Current VPM] / [Metric Bar].
        Returns a single HWC image (uint8).
        """
        # Ensure HWC format is consistent (H, W, C)
        H, W, C = initial_rgb.shape
        
        # Pad between images
        padding_h = 4
        separator = np.full((padding_h, W, C), 200, dtype=np.uint8) # Light grey separator

        # Create the metric bar
        metric_bar = self._metric_bar(phi, utility, step_idx)
        bar_h = metric_bar.shape[0]

        # Assemble the final frame: Initial | Separator | Current | Separator | Metric Bar
        final_frame = np.concatenate(
            [initial_rgb, separator, current_rgb, separator, metric_bar], 
            axis=0
        )
        
        log.debug("Composed frame shape: %s", final_frame.shape)
        return final_frame

    def _save_filmstrip(self, run_dir: Path, frames: List[np.ndarray], steps_meta: List[Dict]) -> None:
        """Writes frame PNGs, the animated GIF, and the JSON metrics to the run directory."""
        
        # 1. Write PNGs for each frame
        frame_dir = run_dir / "frames"
        frame_dir.mkdir(exist_ok=True)
        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(frame_dir / f"frame_{i:02d}.png")
        log.info("Saved %d frames to %s", len(frames), frame_dir)
            
        # 2. Write Animated GIF
        gif_path = run_dir / "filmstrip.gif"
        iio.mimsave(gif_path, frames, duration=0.5, loop=0) # duration is in seconds per frame
        log.info("Saved animated GIF to %s", gif_path)
            
        # 3. Write Metrics JSON
        metrics_path = run_dir / "metrics.json"
        # Use a list of serializable dicts (steps_meta)
        metrics_path.write_text(json.dumps({"steps": steps_meta}, indent=2), encoding="utf-8")
        log.info("Saved metrics JSON to %s", metrics_path)