# stephanie/engine/plan_trace_monitor.py
import os
import json
from datetime import datetime
import time
from typing import Dict, Optional
import traceback

from stephanie.data.plan_trace import PlanTrace, ExecutionStep
from stephanie.agents.plan_trace_scorer import PlanTraceScorerAgent
from stephanie.utils.timing import time_function
from omegaconf import OmegaConf
from stephanie.utils.serialization import default_serializer


class PlanTraceMonitor:
    """Monitors pipeline execution and creates PlanTraces for self-improvement.
    
    This component handles all PlanTrace-related functionality, keeping the Supervisor clean.
    It creates PlanTraces at pipeline start, tracks stage execution, and scores completed traces.
    """

    def __init__(self, cfg: Dict, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.enabled = cfg.get("plan_monitor", {}).get("enabled", False)
        monitor_cfg = cfg.get("plan_monitor", {})
        self.save_output = monitor_cfg.get("save_output", False)
        self.output_dir = monitor_cfg.get("output_dir", "plan_traces")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.current_plan_trace: Optional[PlanTrace] = None
   
        if self.enabled:
            self.plan_trace_scorer = PlanTraceScorerAgent(cfg, memory, logger)
            self.stage_start_times: Dict[int, float] = {}
        
        self.logger.log("PlanTraceMonitorInitialized", {
            "enabled": self.enabled,
            "save_output": self.save_output,
            "cfg_keys": list(cfg.keys())
        })
    
    def start_pipeline(self, context: Dict, pipeline_run_id: str) -> None:
        if not self.enabled:
            self.logger.log("PlanTraceMonitorDisabled", {})
            return 

        """Create PlanTrace when pipeline starts"""
        goal = context.get("goal", {})
        essential_config = {
            k: v for k, v in OmegaConf.to_container(self.cfg, resolve=True).items()
            if k in ["pipeline", "model", "scorer", "dimensions", "scorer_types"]
        }
        
        # Create PlanTrace for this pipeline execution
        self.current_plan_trace = PlanTrace(
            trace_id=str(pipeline_run_id),  # Use pipeline_run_id as trace_id
            goal_id=goal.get("id"),
            goal_text=goal.get("goal_text", ""),
            plan_signature=self._generate_plan_signature(context),
            input_data=self._extract_input_data(context),
            final_output_text="",
            execution_steps=[],
            target_epistemic_quality=None,
            target_epistemic_quality_source=None,
            extra_data={
                "agent_name": "PlanTraceMonitor",
                "started_at": time.time(),
                "pipeline_run_id": pipeline_run_id,
                "pipeline_config": essential_config
            }
        )
        
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
    
    def complete_stage(self, stage_name: str, context: Dict, stage_idx: int) -> None:
        """Update ExecutionStep when stage completes"""
        if not self.current_plan_trace or stage_idx >= len(self.current_plan_trace.execution_steps):
            return
            
        # Calculate duration
        start_time = self.stage_start_times.get(stage_idx, time.time())
        duration = time.time() - start_time
        
        # Update the current step
        step = self.current_plan_trace.execution_steps[stage_idx]
        step.end_time = time.time()
        step.duration = duration
        
        # Capture output preview
        output_keys = list(context.keys())
        output_preview = "Context keys: " + ", ".join(output_keys[:3])
        if len(output_keys) > 3:
            output_preview += f" + {len(output_keys) - 3} more"
        
        step.output_text = output_preview
        step.output_keys = output_keys
        step.output_size = len(str(context))
        
        # Log stage completion
        self.logger.log("PipelineStageCompleted", {
            "trace_id": self.current_plan_trace.trace_id,
            "stage_idx": stage_idx + 1,
            "stage_name": stage_name,
            "stage_time": duration,
            "output_keys": output_keys
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
    
    @time_function()
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
        self.current_plan_trace.extra_data["completed_at"] = time.time()
        
        # Calculate total pipeline time
        start_time = self.current_plan_trace.extra_data.get("started_at", time.time())
        self.current_plan_trace.extra_data["total_time"] = time.time() - start_time
        
        if self.enabled and self.save_output:
            self.save_plan_trace_to_json(self.current_plan_trace)

        # Store in memory
        try:
            self.memory.plan_traces.add(self.current_plan_trace)
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
            "total_time": self.current_plan_trace.extra_data["total_time"]
        })

    @time_function()
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
            
            # Update PlanTrace with scores
            self.current_plan_trace.step_scores = scored_context.get("step_scores", [])
            self.current_plan_trace.pipeline_score = scored_context.get("pipeline_score", {})
            self.current_plan_trace.mars_analysis = scored_context.get("mars_analysis", {})
            
            # Update in memory
            self.memory.plan_traces.update(self.current_plan_trace)
            
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
        self.current_plan_trace.extra_data["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }
        self.current_plan_trace.extra_data["completed_at"] = time.time()
        
        # Store in memory
        try:
            self.memory.plan_traces.add(self.current_plan_trace)
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
