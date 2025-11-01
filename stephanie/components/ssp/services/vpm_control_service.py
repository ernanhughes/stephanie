# stephanie/components/ssp/services/vpm_control_service.py
"""
VPM Control Service - Manages Vectorized Performance Map decision-making for SSP components

This service provides the core control loop for the Self-Play System (SSP), using VPM (Vectorized 
Performance Map) methodology to make decisions about when to continue refining, resample, escalate,
or stop processing based on multi-dimensional performance metrics.

Key features:
- Integrates with the VPMController for trend-aware decision making
- Manages state for multiple processing units (questions, episodes, etc.)
- Generates VPM visualization artifacts for monitoring and analysis
- Supports bandit-based exemplar selection for adaptive refinement
- Provides goal-aware stopping conditions based on configured targets

The service follows the Service Protocol interface for integration with the Stephanie framework.
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from stephanie.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.services.service_protocol import Service

# Import VPM core components from the examples provided
from stephanie.zeromodel.vpm_controller import (
    Thresholds, 
    Policy, 
    VPMRow, 
    Decision,
    VPMController as CoreVPMController
)
from stephanie.zeromodel.vpm_phos import (
    build_vpm_phos_artifacts
)

class VPMControlService(Service):
    """
    Service for managing VPM-based control decisions within the SSP framework.
    
    This service:
    - Creates and manages VPMControllers for individual processing units
    - Makes decisions based on multi-dimensional performance metrics
    - Generates VPM visualization artifacts for monitoring
    - Integrates with bandit systems for adaptive exemplar selection
    - Provides goal-aware stopping conditions
    
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
        Initialize the VPM Control Service.
        
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
        self._controllers: Dict[str, CoreVPMController] = {}
        self._bus = getattr(memory, "bus", None)
        
        # Configuration setup
        self._setup_configuration()
        
        # Visualization configuration
        self._setup_visualization_paths()
        
        # Bandit integration
        self._bandit_choose: Optional[Callable[[List[str]], str]] = None
        self._bandit_update: Optional[Callable[[str, float], None]] = None
        
        # Metrics tracking
        self._metrics_history: Dict[str, List[Dict[str, float]]] = {}

    def _setup_configuration(self) -> None:
        """Configure thresholds, policies, and parameters from config."""
        c = (self.cfg.get("vpm_control") or {})
        
        # Code thresholds (for code-related processing)
        self._thr_code = Thresholds(
            mins=c.get("mins_code", {
                "tests_pass_rate": 1.0,
                "coverage": 0.70,
                "type_safe": 1.0,
                "lint_clean": 1.0,
                "complexity_ok": 0.8,
            }),
            stop_margin=float(c.get("stop_margin_code", 0.0)),
            edit_margin=float(c.get("edit_margin_code", 0.0)),
        )
        
        # Text thresholds (for question/answer processing)
        self._thr_text = Thresholds(
            mins=c.get("mins_text", {
                "coverage": 0.80,
                "correctness": 0.75,
                "coherence": 0.75,
                "citation_support": 0.65,
                "entity_consistency": 0.80,
            }),
            stop_margin=float(c.get("stop_margin_text", 0.02)),
            edit_margin=float(c.get("edit_margin_text", 0.01)),
        )
        
        # Policy configuration
        self._policy = Policy(
            window=int(c.get("window", 5)),
            ema_alpha=float(c.get("ema_alpha", 0.4)),
            edit_margin=float(c.get("edit_margin", 0.05)),
            patience=int(c.get("patience", 3)),
            escalate_after=int(c.get("escalate_after", 2)),
            oscillation_window=int(c.get("oscillation_window", 6)),
            oscillation_threshold=int(c.get("oscillation_threshold", 3)),
            cooldown_steps=int(c.get("cooldown_steps", 1)),
            spinoff_dim=str(c.get("spinoff_dim", "novelty")),
            stickiness_dim=str(c.get("stickiness_dim", "stickiness")),
            spinoff_gate=tuple(c.get("spinoff_gate", (0.75, 0.45))),
            max_regressions=int(c.get("max_regressions", 2)),
            zscore_clip_dims=list(c.get("zscore_clip_dims", [
                "coverage", "coherence", "correctness", "tests_pass_rate"
            ])),
            zscore_clip_sigma=float(c.get("zscore_clip_sigma", 3.5)),
            local_gap_dims=list(c.get("local_gap_dims", [
                "citation_support", "entity_consistency", "lint_clean", "type_safe"
            ])),
            max_steps=int(c.get("max_steps", 50)),
            goal_kind=c.get("goal_kind"),
            goal_name=c.get("goal_name"),
            goal_min_score=float(c.get("goal_min_score", 0.75)),
            goal_allow_unmet=int(c.get("goal_allow_unmet", 0)),
        )

    def _setup_visualization_paths(self) -> None:
        """Configure paths for VPM visualization artifacts."""
        c = (self.cfg.get("vpm_control") or {})
        
        # Base directory for visualization artifacts
        self._viz_dir = c.get("viz_dir", "./runs/vpm_visualizations")
        os.makedirs(self._viz_dir, exist_ok=True)
        
        # Subdirectories for different visualization types
        self._raw_viz_dir = os.path.join(self._viz_dir, "raw")
        self._phos_viz_dir = os.path.join(self._viz_dir, "phos")
        self._compare_viz_dir = os.path.join(self._viz_dir, "comparison")
        
        os.makedirs(self._raw_viz_dir, exist_ok=True)
        os.makedirs(self._phos_viz_dir, exist_ok=True)
        os.makedirs(self._compare_viz_dir, exist_ok=True)
        
        # TL fractions for PHOS guard sweep
        self._tl_fracs = c.get("tl_fracs", [0.25, 0.16, 0.36, 0.09])
        self._delta = c.get("delta", 0.02)
        
        # Dimensions to visualize
        self._dimensions = c.get("dimensions", [
            "coverage", "correctness", "coherence", "citation_support", "entity_consistency"
        ])

    # ---------------- Service Protocol Implementation ----------------
    
    def initialize(self, **kwargs) -> None:
        """Initialize the service and required resources."""
        if self._initialized:
            return
            
        self._initialized = True
        self.logger.log("VPMControlServiceInit", {
            "viz_dir": self._viz_dir,
            "tl_fracs": self._tl_fracs,
            "delta": self._delta,
            "dimensions": self._dimensions
        })

    def shutdown(self) -> None:
        """Clean up resources and persist state if needed."""
        # Persist any remaining controller state
        for unit, controller in self._controllers.items():
            try:
                controller._persist_state()
            except Exception as e:
                self.logger.log("VPMControlServiceWarning", {
                    "event": "state_persist_failed",
                    "unit": unit,
                    "error": str(e)
                })
                
        self._controllers.clear()
        self._initialized = False
        self.logger.log("VPMControlServiceShutdown", {})

    def health_check(self) -> Dict[str, Any]:
        """Return service health status and metrics."""
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "units": len(self._controllers),
            "timestamp": time.time(),
            "viz_dir": self._viz_dir,
            "active_dimensions": self._dimensions
        }

    @property
    def name(self) -> str:
        """Service name for identification."""
        return "vpm-control-service"

    # ---------------- Public API ----------------
    
    def decide(
        self,
        unit: str,
        *,
        kind: str,
        dims: Dict[str, float],
        step_idx: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
        candidate_exemplars: Optional[List[str]] = None,
    ) -> Decision:
        """
        Make a VPM-based control decision for a processing unit.
        
        Args:
            unit: Identifier for the processing unit (e.g., question ID)
            kind: Type of processing ("text" or "code")
            dims: Performance metrics for the current state
            step_idx: Current step index in the processing pipeline
            meta: Additional metadata for decision context
            candidate_exemplars: Available exemplars for resampling
            
        Returns:
            Decision object with signal, reason, parameters, and snapshot
        """
        # Get or create controller for this unit
        ctrl = self._get_controller(unit)
        
        # Create VPM row for this state
        m = dict(meta or {})
        if candidate_exemplars:
            m["candidate_exemplars"] = candidate_exemplars
            
        row = VPMRow(
            unit=unit,
            kind=("code" if kind == "code" else "text"),
            timestamp=time.time(),
            step_idx=step_idx,
            dims={k: float(v) for k, v in dims.items()},
            meta=m,
        )
        
        # Track metrics history for visualization
        self._track_metrics(unit, dims, step_idx)
        
        # Make decision
        dec = ctrl.add(row, candidate_exemplars=candidate_exemplars)
        
        # Emit and persist decision
        self._publish(unit, dec)
        # self._persist_trace(unit, row, dec)
        
        # Generate visualization if needed
        if step_idx is not None and step_idx % 5 == 0:  # Every 5 steps
            self.generate_visualization(unit, step_idx)
            
        return dec

    def decide_many(self, unit: str, frames: List[dict]) -> List[Decision]:
        """
        Process multiple metric frames for a single unit (batch processing).
        
        Args:
            unit: Identifier for the processing unit
            frames: List of metric frames to process
            
        Returns:
            List of decisions corresponding to each frame
        """
        out = []
        for f in frames:
            out.append(self.decide(
                unit,
                kind=f.get("kind", "text"),
                dims=f.get("dims", {}),
                step_idx=f.get("step_idx"),
                meta=f.get("meta"),
                candidate_exemplars=f.get("candidate_exemplars"),
            ))
        return out

    def reset_unit(self, unit: str) -> None:
        """Reset the controller state for a specific unit."""
        if unit in self._controllers:
            del self._controllers[unit]
            self.logger.log("VPMControlResetUnit", {"unit": unit})

    def get_unit_state(self, unit: str) -> Dict[str, Any]:
        """
        Get the current state of a unit's controller.
        
        Returns:
            Dictionary with controller state information
        """
        ctrl = self._controllers.get(unit)
        if not ctrl:
            return {}
            
        return {
            "resample_counts": dict(ctrl.resample_counts),
            "cooldown_until_step": dict(ctrl.cooldown_until_step),
            "last_signal": {k: v.name for k, v in ctrl.last_signal.items()},
        }

    def set_goal_gate(
        self, 
        *, 
        goal_kind: Optional[str], 
        goal_name: Optional[str],
        min_score: float = 0.75, 
        allow_unmet: int = 0
    ) -> None:
        """
        Update goal-aware parameters for all controllers.
        
        Args:
            goal_kind: Type of goal ("text" or "code")
            goal_name: Name of the specific goal
            min_score: Minimum score required to meet goal
            allow_unmet: Number of unmet dimensions allowed
        """
        self._policy.goal_kind = goal_kind
        self._policy.goal_name = goal_name
        self._policy.goal_min_score = float(min_score)
        self._policy.goal_allow_unmet = int(allow_unmet)
        
        # Update existing controllers
        for ctrl in self._controllers.values():
            ctrl.p = self._policy
            
        self.logger.log("VPMControlGoalGateUpdated", {
            "goal_kind": goal_kind,
            "goal_name": goal_name,
            "min_score": min_score,
            "allow_unmet": allow_unmet
        })

    def attach_bandit(
        self, 
        choose_fn: Callable[[List[str]], str], 
        update_fn: Callable[[str, float], None]
    ) -> None:
        """
        Attach bandit hooks for adaptive exemplar selection.
        
        Args:
            choose_fn: Function to select exemplar from candidates
            update_fn: Function to update bandit with reward
        """
        self._bandit_choose = choose_fn
        self._bandit_update = update_fn
        
        # Update existing controllers
        for ctrl in self._controllers.values():
            ctrl.bandit_choose = choose_fn
            ctrl.bandit_update = update_fn
            
        self.logger.log("VPMControlBanditAttached", {})

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
            tl_frac=0.25,  # Default for single visualization
            interleave=False,
            weights=None,
        )
        
        # Generate comparison artifacts if we have multiple models (for future extension)
        # This would require tracking multiple models' performance
        
        return {
            "raw": artifacts["paths"]["raw"],
            "phos": artifacts["paths"]["phos"],
            "metrics": json.dumps(artifacts["metrics"]),
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
        # This would be implemented when comparing different SSP configurations
        # For now, it's a placeholder for future extension
        if output_path is None:
            output_path = os.path.join(
                self._compare_viz_dir, 
                f"{unit.replace(':', '_')}_{model_a}_vs_{model_b}"
            )
            
        # In a real implementation, we would:
        # 1. Gather metrics for both models
        # 2. Build a DataFrame with both models' performance
        # 3. Call build_compare_guarded
        
        # Placeholder implementation
        return {
            "message": "Comparison visualization not yet implemented for this service"
        }

    # ---------------- Internal Helpers ----------------
    
    def _get_controller(self, unit: str) -> CoreVPMController:
        """Get or create a VPM controller for the specified unit."""
        if unit not in self._controllers:
            # Create state path for this unit
            state_path = os.path.join(
                self._viz_dir, 
                f"vpm_state_{unit.replace(':', '_')}.json"
            )
            
            # Create controller
            ctrl = CoreVPMController(
                thresholds_code=self._thr_code,
                thresholds_text=self._thr_text,
                policy=self._policy,
                bandit_choose=self._bandit_choose,
                bandit_update=self._bandit_update,
                logger=lambda ev, d: self.logger.log(ev, d),
                state_path=state_path,
            )
            
            self._controllers[unit] = ctrl
            
        return self._controllers[unit]

    def _publish(self, unit: str, dec: Decision) -> None:
        """Publish decision to event bus if available."""
        try:
            if self._bus and hasattr(self._bus, "publish"):
                self._bus.publish("vpm.control.decision", {
                    "unit": unit, 
                    **asdict(dec)
                })
        except Exception as e:
            self.logger.log("VPMControlServiceWarning", {
                "event": "publish_failed",
                "unit": unit,
                "error": str(e)
            })

    def _persist_trace(self, unit: str, row: VPMRow, dec: Decision) -> None:
        """Persist decision trace to memory storage."""
        try:
            repo = getattr(self.memory, "plan_traces", None) or getattr(self.memory, "traces", None)
            if repo and hasattr(repo, "insert"):
                repo.insert({
                    "ts": time.time(),
                    "kind": "vpm_control_decision",
                    "unit": unit,
                    "signal": dec.signal.name,
                    "reason": dec.reason,
                    "params": dec.params,
                    "snapshot": dec.snapshot,
                    "metrics": row.dims,
                    "step_idx": row.step_idx,
                })
        except Exception as e:
            self.logger.log("VPMControlServiceWarning", {
                "event": "trace_persist_failed",
                "unit": unit,
                "error": str(e)
            })

    def _track_metrics(
        self, 
        unit: str, 
        dims: Dict[str, float], 
        step_idx: Optional[int]
    ) -> None:
        """Track metrics history for visualization purposes."""
        if unit not in self._metrics_history:
            self._metrics_history[unit] = []
            
        # Create a record with step index and metrics
        record = {
            "step_idx": step_idx or len(self._metrics_history[unit]),
            **{k: float(v) for k, v in dims.items()}
        }
        
        self._metrics_history[unit].append(record)
        
        # Keep history bounded
        max_history = self.cfg.get("vpm_control", {}).get("max_metrics_history", 100)
        if len(self._metrics_history[unit]) > max_history:
            self._metrics_history[unit] = self._metrics_history[unit][-max_history:]

    def _convert_to_dataframe(self, metrics_history: List[Dict]) -> Any:
        """
        Convert metrics history to DataFrame format expected by VPM builder.
        
        In a real implementation, this would create a proper pandas DataFrame.
        For this example, we'll create a simplified structure that matches expectations.
        """
        try:
            import pandas as pd
            # Create a DataFrame with multi-index for model and dimension
            df_data = []
            for record in metrics_history:
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
            return df
        except ImportError:
            # Fallback for environments without pandas
            # This would be a minimal implementation just for the example
            return {
                "node_id": [str(i) for i in range(len(metrics_history))],
                **{dim: [] for dim in self._dimensions}
            }

    # ---------------- Debugging ----------------
    
    def __repr__(self):
        """String representation for debugging."""
        active_count = len(self._controllers)
        return (
            f"<VPMControlService: status={'initialized' if self._initialized else 'uninitialized'}  "
            f"units={active_count}  "
            f"dimensions={len(self._dimensions)}>"
        )