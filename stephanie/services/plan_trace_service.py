# stephanie/engine/plan_trace_monitor.py
import json
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from stephanie.agents.agent_scorer import AgentScorerAgent
from stephanie.agents.plan_trace_scorer import PlanTraceScorerAgent
from stephanie.constants import PLAN_TRACE_ID, SCORABLE_DETAILS
from stephanie.data.plan_trace import ExecutionStep, PlanTrace
from stephanie.models.plan_trace import PlanTraceORM
from stephanie.scoring.scorable import ScorableFactory
from stephanie.services.service_protocol import Service
from stephanie.utils.serialization import default_serializer


class PlanTraceService(Service):
    """
    Service that monitors pipeline execution and creates PlanTraces for self-improvement.

    Responsibilities:
    - Create PlanTraces at pipeline start
    - Track stage execution and completion
    - Score completed traces
    - Apply retention policies
    """

    def __init__(self, cfg: Dict, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.container = container

        monitor_cfg = cfg.get("plan_monitor", {})
        self.enabled = monitor_cfg.get("enabled", False)
        self.save_output = monitor_cfg.get("save_output", False)
        self.output_dir = monitor_cfg.get("output_dir", "plan_traces")
        self.retention_policy = monitor_cfg.get("retention_policy", "keep_all")

        if self.save_output and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.current_plan_trace: Optional[PlanTrace] = None
        self.reuse_links = []  # (parent_trace_id, child_trace_id)
        self.stage_start_times: Dict[int, float] = {}

        if self.enabled:
            self.plan_trace_scorer = PlanTraceScorerAgent(cfg, memory, container, logger)
            self.plan_trace_scorer.container = self.container
            self.agent_scorer = AgentScorerAgent(cfg, memory, container, logger)
            self.agent_scorer.container = self.container
    
        self._initialized = False

    # === Service Protocol ===
    def initialize(self, **kwargs) -> None:
        self._initialized = True
        if self.logger:
            self.logger.log("PlanTraceServiceInit", {"status": "initialized", "enabled": self.enabled})

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "enabled": self.enabled,
                "current_trace": bool(self.current_plan_trace),
                "reuse_links": len(self.reuse_links),
                "stages_tracked": len(self.stage_start_times),
            },
            "dependencies": {
                "scoring_service": "attached" if hasattr(self, "plan_trace_scorer") else "missing"
            },
        }

    def shutdown(self) -> None:
        """Reset and disable service."""
        self.current_plan_trace = None
        self.stage_start_times = {}
        self._initialized = False
        if self.logger:
            self.logger.log("PlanTraceServiceShutdown", {})

    @property
    def name(self) -> str:
        return "plan-trace-v1"

    # === Domain Logic (unchanged from your version, except now under Service) ===
    def link_reuse(self, parent_trace_id: str, child_trace_id: str):
        self.memory.plan_traces.add_reuse_link(parent_trace_id, child_trace_id)

    def revise_trace(self, trace_id: str, revision: dict):
        trace: PlanTraceORM = self.memory.plan_traces.get_by_trace_id(trace_id)
        if not trace:
            return
        trace.meta.setdefault("revisions", []).append({
            "timestamp": time.time(),
            **revision
        })
        self.memory.plan_traces.upsert(trace)
        self.logger.log("PlanTraceRevised", {"trace_id": trace_id, "revision": revision})

    def apply_retention_policy(self):
        if self.retention_policy == "keep_all":
            return
        elif self.retention_policy == "keep_top_k":
            top_k = self.cfg["plan_monitor"].get("retain_k", 100)
            traces = self.memory.plan_traces.get_all()
            sorted_traces = sorted(traces, key=lambda t: t.pipeline_score.get("overall", 0), reverse=True)
            for t in sorted_traces[top_k:]:
                self.memory.plan_traces.delete(t.trace_id)
                self.logger.log("PlanTraceDiscarded", {"trace_id": t.trace_id})
        elif self.retention_policy == "discard_failed":
            failed = self.memory.plan_traces.get_failed()
            for t in failed:
                self.memory.plan_traces.delete(t.trace_id)
                self.logger.log("PlanTraceDiscarded", {"trace_id": t.trace_id, "reason": "failed"})

    def start_pipeline(self, context: Dict, pipeline_run_id: str) -> None:
        if not self.enabled:
            self.logger.log("PlanTraceMonitorDisabled", {})
            return 

        """Create PlanTrace when pipeline starts"""
        goal = context.get("goal", {})
        essential_config = {
            k: v for k, v in OmegaConf.to_container(self.cfg, resolve=True).items()
            if k in ["pipeline", "model", "scorer", "dimensions", "enabled_scorers"]
        }
        
        # Create PlanTrace for this pipeline execution
        self.current_plan_trace = PlanTrace(
            trace_id=str(pipeline_run_id),  # Use pipeline_run_id as trace_id
            pipeline_run_id=pipeline_run_id,
            goal_id=goal.get("id"),
            goal_text=goal.get("goal_text", ""),
            plan_signature=self._generate_plan_signature(context),
            input_data=self._extract_input_data(context),
            final_output_text="",
            execution_steps=[],
            target_epistemic_quality=None,
            target_epistemic_quality_source=None,
            meta={
                "agent_name": "PlanTraceMonitor",
                "started_at": time.time(),
                "pipeline_run_id": pipeline_run_id,
                "pipeline_config": essential_config
            }
        )

        # After creating self.current_plan_trace
        self.memory.plan_traces.upsert(self.current_plan_trace)
        context[PLAN_TRACE_ID] = self.current_plan_trace.trace_id

        
        # Log PlanTrace creation
        self.logger.log("PlanTraceCreated", {
            "trace_id": pipeline_run_id,
            "goal_id": goal.get("id"),
            "goal_text": (goal.get("goal_text", "")[:100] + "...") if goal.get("goal_text") else None
        })
    
    def _generate_plan_signature(self, context: Dict) -> str:
        """Generate a signature identifying this pipeline configuration"""
        pipeline = context.get("pipeline", [])
        return f"{'_'.join(pipeline)}"
    
    def _extract_input_data(self, context: Dict) -> Dict:
        """Extract relevant input data for the PlanTrace"""
        # Only capture essential input data, not the entire context
        return {
            "input_keys": list(context.keys()),
            "goal_id": context.get("goal", {}).get("id"),
            "goal_text_preview": (context.get("goal", {}).get("goal_text", "")[:100] + "...")
                if context.get("goal", {}).get("goal_text") else None
        }
    
    def start_stage(self, stage_name: str, context: Dict, stage_idx: int) -> None:
        """Create ExecutionStep when stage starts"""
        if not self.current_plan_trace:
            return
            
        # Record start time
        self.stage_start_times[stage_idx] = time.time()
        
        # Create step ID
        step_id = f"{self.current_plan_trace.trace_id}_step_{stage_idx + 1}"
        
        # Create step description
        description = f"Stage {stage_idx + 1}: {stage_name}"
        
        # Extract input data (simplified)
        input_preview = "Context keys: " + ", ".join(list(context.keys())[:3])
        if len(context.keys()) > 3:
            input_preview += f" + {len(context.keys()) - 3} more"
        
        # Create ExecutionStep
        execution_step = ExecutionStep(
            step_id=step_id,
            pipeline_run_id=self.current_plan_trace.pipeline_run_id,
            step_order=stage_idx + 1,
            step_type=stage_name,
            description=description,
            input_text=input_preview,
            output_text="",
            agent_name=stage_name,
            start_time=time.time(),
            error=None,
            scores=None
        )
        
        # Add to PlanTrace
        self.current_plan_trace.execution_steps.append(execution_step)
        
        # Log stage start
        self.logger.log("PipelineStageStarted", {
            "trace_id": self.current_plan_trace.trace_id,
            "stage_idx": stage_idx + 1,
            "stage_name": stage_name
        })
    
    async def complete_stage(
        self,
        stage_name: str,
        context: Dict,
        stage_idx: int,
    ) -> None:
        """Update ExecutionStep when stage completes."""
        if not self.current_plan_trace or stage_idx >= len(self.current_plan_trace.execution_steps):
            return

        # Calculate duration
        start_time = self.stage_start_times.get(stage_idx, time.time())
        duration = time.time() - start_time

        # Update the current step
        step = self.current_plan_trace.execution_steps[stage_idx]
        step.end_time = time.time()
        step.duration = duration

        scorable_details = context.get(SCORABLE_DETAILS, {})
        if scorable_details.get("output_text"):
            step.input_text = scorable_details.get("input_text", "")
            step.output_text = scorable_details.get("output_text", "")
            step.description = scorable_details.get(
                "description", f"Scorable output from {stage_name}"
            )

            # Kick scoring agent
            context = await self.agent_scorer.run(context)
        else:
            # Fallback breadcrumb
            output_keys = list(context.keys())
            preview = "Context keys: " + ", ".join(output_keys[:3])
            if len(output_keys) > 3:
                preview += f" + {len(output_keys) - 3} more"
            step.output_text = preview
            step.output_keys = output_keys
            step.output_size = len(str(context))

        self.logger.log("PipelineStageCompleted", {
            "trace_id": self.current_plan_trace.trace_id,
            "stage_idx": stage_idx + 1,
            "stage_name": stage_name,
            "stage_time": duration,
            "is_scorable": bool(scorable_details)
        })
    
    def handle_stage_error(self, stage_name: str, error: Exception, stage_idx: int) -> None:
        """Update ExecutionStep when stage errors"""
        if not self.current_plan_trace or stage_idx >= len(self.current_plan_trace.execution_steps):
            return
            
        # Calculate duration
        start_time = self.stage_start_times.get(stage_idx, time.time())
        duration = time.time() - start_time
        
        # Update the current step with error information
        step = self.current_plan_trace.execution_steps[stage_idx]
        step.end_time = time.time()
        step.duration = duration
        step.error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }
        
        # Log error
        self.logger.log("PipelineStageError", {
            "trace_id": self.current_plan_trace.trace_id,
            "stage_idx": stage_idx + 1,
            "stage_name": stage_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stage_duration": duration
        })
    
    async def complete_pipeline(self, context: Dict) -> None:
        """Complete the PlanTrace when pipeline ends"""
        if not self.current_plan_trace:
            return
            
        # Set final output text
        final_output = context.get("final_output", "")
        if isinstance(final_output, str):
            self.current_plan_trace.final_output_text = (
                final_output[:1000] + "..." if len(final_output) > 1000 else final_output
            )
        elif isinstance(final_output, dict):
            self.current_plan_trace.final_output_text = str(final_output)[:1000] + "..."
        else:
            self.current_plan_trace.final_output_text = str(final_output)[:1000] + "..."
        
        # Set completion time
        self.current_plan_trace.meta["completed_at"] = time.time()
        
        # Calculate total pipeline time
        start_time = self.current_plan_trace.meta.get("started_at", time.time())
        self.current_plan_trace.meta["total_time"] = time.time() - start_time
        
        if self.enabled and self.save_output:
            self.save_plan_trace_to_json(self.current_plan_trace)

        # Store in memory
        try:
            self.memory.plan_traces.upsert(self.current_plan_trace)
            scorable = ScorableFactory.from_orm(self.current_plan_trace)
            self.memory.scorable_embeddings.get_or_create(scorable)
            self.logger.log("PlanTraceStored", {
                "trace_id": self.current_plan_trace.trace_id,
                "step_count": len(self.current_plan_trace.execution_steps)
            })
        except Exception as e:
            self.logger.log("PlanTraceStorageError", {
                "trace_id": self.current_plan_trace.trace_id,
                "error": str(e)
            })
        
        self.logger.log("PlanTraceCompleted", {
            "trace_id": self.current_plan_trace.trace_id,
            "step_count": len(self.current_plan_trace.execution_steps),
            "total_time": self.current_plan_trace.meta["total_time"]
        })

    async def score_pipeline(self, context: Dict) -> None:
        """Score the completed PlanTrace"""
        if not self.current_plan_trace:
            return
            
        try:
            # Run PlanTraceScorerAgent
            scoring_context = {
                "plan_traces": [self.current_plan_trace],
                "goal": context.get("goal", {})
            }
            
            # Score the PlanTrace
            scored_context = await self.plan_trace_scorer.run(scoring_context)
            
            if not scored_context:
                self.logger.log("PlanTraceScoringWarning", {
                    "trace_id": self.current_plan_trace.trace_id,
                    "reason": "scorer returned None"
                })
                return  # or keep current_plan_trace unscored

            # Update PlanTrace with scores
            self.current_plan_trace.step_scores = scored_context.get("step_scores", [])
            self.current_plan_trace.pipeline_score = scored_context.get("pipeline_score", {})
            self.current_plan_trace.mars_analysis = scored_context.get("mars_analysis", {})
            
            # Update in memory
            self.memory.plan_traces.upsert(self.current_plan_trace)
            
            self.logger.log("PlanTraceScored", {
                "trace_id": self.current_plan_trace.trace_id,
                "step_count": len(self.current_plan_trace.execution_steps),
                "pipeline_score": scored_context.get("pipeline_score", {})
            })
        except Exception as e:
            self.logger.log("PlanTraceScoringError", {
                "trace_id": self.current_plan_trace.trace_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def handle_pipeline_error(self, error: Exception, context: Dict) -> None:
        """Handle errors that occur during pipeline execution"""
        if not self.current_plan_trace:
            return
            
        # Update PlanTrace with error information
        self.current_plan_trace.final_output_text = f"Pipeline failed: {str(error)}"
        self.current_plan_trace.meta["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }
        self.current_plan_trace.meta["completed_at"] = time.time()
        
        # Store in memory
        try:
            self.memory.plan_traces.upsert(self.current_plan_trace)
        except Exception as e:
            self.logger.log("PlanTraceSaveError", {
                "trace_id": self.current_plan_trace.trace_id,
                "error": str(e)
            })
        
        self.logger.log("PlanTraceError", {
            "trace_id": self.current_plan_trace.trace_id,
            "error_type": type(error).__name__,
            "error_message": str(error)
        })
    
    def reset(self) -> None:
        """Reset the monitor for the next pipeline"""
        self.current_plan_trace = None
        self.stage_start_times = {}

    def save_plan_trace_to_json(self, plan_trace: PlanTrace) -> str:
        """Save PlanTrace to JSON file in the output directory"""
        if not self.enabled or not self.output_dir:
            return ""
        
        try:
            # Create filename with trace_id and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{plan_trace.trace_id}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Convert to dictionary and save with custom serializer
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    plan_trace.to_dict(), 
                    f, 
                    indent=2, 
                    default=default_serializer,
                    ensure_ascii=False
                )
            
            print(f"✅ PlanTrace saved to: {filepath}")
            self.logger.log("PlanTraceSavedToFile", {
                "trace_id": plan_trace.trace_id,
                "filepath": filepath,
                "step_count": len(plan_trace.execution_steps)
            })
            
            return filepath
        except Exception as e:
            self.logger.log("PlanTraceSaveToFileError", {
                "trace_id": plan_trace.trace_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
