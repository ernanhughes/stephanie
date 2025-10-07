# stephanie/services/zeromodel_service.py
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from stephanie.services.service_protocol import Service
from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.tools.gif_logger import GifLogger

_DEFAULT_PIPELINE = [
    {"name": "normalize", "params": {}},
    {"name": "feature_engineering", "params": {}},
    {"name": "organization", "params": {"strategy": "spatial"}},
    {"name": "timeline_renderer", "params": {
        "show_line": True,
        "show_bars": True,
        "bar_alpha": 0.3,
        "line_width": 2.0,
        "x_label": "Time (seconds)",
        "y_label": "Metric Value"
    }}
]

class ZeroModelService(Service):
    """
    Single service for all zero model operations with agent-controlled timelines.
    - No global event subscriptions
    - Agents explicitly create timelines via RPC
    - Timeline contexts are isolated by run_id
    - Full control over event processing and rendering
    """
    
    def __init__(self, cfg: Dict[str, Any], memory, logger):
        super().__init__(cfg, memory, logger)
        self.timeline_contexts: Dict[str, List[Dict]] = {}  # run_id -> event list
        self.fps = int(cfg.get("zero_model", {}).get("fps", 6))
        self.max_frames = int(cfg.get("zero_model", {}).get("max_frames", 512))
        self.output_dir = cfg.get("zero_model", {}).get("output_dir", "reports/vpm")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize pipeline
        pipeline_cfg = cfg.get("zero_model", {}).get("pipeline") or _DEFAULT_PIPELINE
        self.pipeline = PipelineExecutor(pipeline_cfg)
        self._initialized = True

    def initialize(self, **kwargs) -> None:
        pass  # Already initialized in __init__

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "active_timelines": len(self.timeline_contexts),
                "fps": self.fps,
                "max_frames": self.max_frames
            }
        }

    def shutdown(self) -> None:
        self.timeline_contexts.clear()
        self._initialized = False

    @property
    def name(self) -> str:
        return "zeromodel-service-v1"

    # --- Agent-controlled timeline RPC handlers ---
    def rpc_routes(self) -> Dict[str, Any]:
        return {
            "create_timeline": self._create_timeline,
            "add_event": self._add_event,
            "finalize_timeline": self._finalize_timeline,
            "get_timeline_status": self._get_timeline_status
        }

    async def _create_timeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        run_id = payload.get("run_id", str(uuid.uuid4()))
        if run_id in self.timeline_contexts:
            return {"error": f"Timeline already exists for run_id: {run_id}"}
        
        self.timeline_contexts[run_id] = {
            "events": [],
            "started_at": time.time(),
            "status": "active",
            "config": payload.get("config", {})
        }
        return {"run_id": run_id, "status": "created"}

    async def _add_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        run_id = payload.get("run_id")
        if not run_id or run_id not in self.timeline_contexts:
            return {"error": "Invalid run_id or timeline not created"}
        
        event = payload.get("event")
        if not event:
            return {"error": "Missing event payload"}
            
        self.timeline_contexts[run_id]["events"].append(event)
        return {"run_id": run_id, "event_count": len(self.timeline_contexts[run_id]["events"])}

    async def _finalize_timeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        run_id = payload.get("run_id")
        if not run_id or run_id not in self.timeline_contexts:
            return {"error": "Invalid run_id or timeline not created"}
        
        context = self.timeline_contexts[run_id]
        try:
            # Build timeline matrix from events
            matrix = self._build_timeline_matrix(context["events"])
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base_name = f"run_{run_id[:8]}_{timestamp}"
            out_path = os.path.join(self.output_dir, f"{base_name}.gif")
            
            # Render timeline using ZeroModel pipeline
            result = self._render_timeline(
                matrix,
                out_path,
                metrics=context["config"].get("metrics", ["metric", "draft", "improve", "debug"]),
                options=context["config"].get("options", {})
            )
            
            # Emit timeline_ready event
            await self.memory.bus.publish("events.zero_model.timeline_ready", {
                "run_id": run_id,
                "timeline_path": result["output_path"],
                "frames": result["frames"],
                "shape": result["shape"],
                "metric_summary": self._calculate_summary(context["events"])
            })
            
            # Clean up context
            del self.timeline_contexts[run_id]
            return {
                "status": "success",
                "timeline_path": result["output_path"],
                "frames": result["frames"],
                "summary": self._calculate_summary(context["events"])
            }
        except Exception as e:
            # Keep context for debugging
            context["status"] = "failed"
            context["error"] = str(e)
            return {"status": "error", "error": str(e)}

    async def _get_timeline_status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        run_id = payload.get("run_id")
        if not run_id or run_id not in self.timeline_contexts:
            return {"status": "not_found"}
        
        context = self.timeline_contexts[run_id]
        return {
            "run_id": run_id,
            "status": context["status"],
            "event_count": len(context["events"]),
            "started_at": context["started_at"],
            "last_event": context["events"][-1] if context["events"] else None
        }

    # --- Internal helpers ---
    def _build_timeline_matrix(self, events: List[Dict]) -> np.ndarray:
        """Convert events to timeline matrix for visualization"""
        n = len(events)
        matrix = np.zeros((n, 4), dtype=np.float32)
        
        for i, event in enumerate(events):
            # Metric value (primary Y-axis)
            matrix[i, 0] = float(event.get("metric", 0.0))
            
            # Node type distribution (stacked bars)
            node_type = event.get("node_type", "unknown")
            if node_type == "draft":
                matrix[i, 1] = 1.0
            elif node_type == "improve":
                matrix[i, 2] = 1.0
            elif node_type == "debug":
                matrix[i, 3] = 1.0
                
        return matrix

    def _render_timeline(self, matrix: np.ndarray, out_path: str, 
                        metrics: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        """Render timeline using ZeroModel pipeline"""
        # Apply default options
        default_options = {
            "show_line": True,
            "show_bars": True,
            "bar_alpha": 0.3,
            "line_width": 2.0,
            "x_label": "Time (seconds)",
            "y_label": "Metric Value",
            "title": "Prompt Search Timeline"
        }
        options = {**default_options, **options}
        
        # Configure timeline renderer stage
        for stage in self.pipeline.stages:
            if stage.name == "timeline_renderer":
                stage.params.update(options)
                break
        
        ctx = {
            "vpm": matrix,
            "metric_names": metrics,
            "gif_logger": GifLogger(max_frames=self.max_frames),
            "fps": self.fps
        }
        
        self.pipeline.process(ctx)
        ctx["gif_logger"].save_gif(out_path, fps=ctx["fps"])
        
        return {
            "output_path": out_path,
            "frames": len(ctx["gif_logger"].frames),
            "shape": list(matrix.shape)
        }

    def _calculate_summary(self, events: List[Dict]) -> Dict[str, Any]:
        """Calculate timeline metrics summary"""
        if not events:
            return {}
            
        metrics = [float(e.get("metric", 0.0)) for e in events]
        return {
            "total_nodes": len(events),
            "best_metric": max(metrics) if metrics else 0.0,
            "avg_metric": sum(metrics) / len(metrics) if metrics else 0.0,
            "draft_count": sum(1 for e in events if e.get("node_type") == "draft"),
            "improve_count": sum(1 for e in events if e.get("node_type") == "improve"),
            "debug_count": sum(1 for e in events if e.get("node_type") == "debug"),
            "buggy_count": sum(1 for e in events if e.get("buggy", False))
        }