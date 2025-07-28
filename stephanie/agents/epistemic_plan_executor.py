# stephanie/agents/epistemic_plan_executor_agent.py
import os
import traceback
from typing import Any, Dict, List, Optional
import time
import uuid

from stephanie.scoring.scorable_factory import TargetType, ScorableFactory
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.data.plan_trace import PlanTrace, ExecutionStep
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.hrm_scorer import HRMScorer  # Optional: for future use

import dspy

class EpistemicPlanExecutorAgent(BaseAgent):
    """
    Agent to execute a reasoning plan (e.g., a DSPy program) and generate
    a detailed PlanTrace for subsequent analysis by the Epistemic Plan HRM.
    """

    def __init__(
        self, cfg: Dict[str, Any], memory: Any = None, logger: Any = None
    ):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", [])
        self.plan_timeout_seconds = cfg.get("plan_timeout_seconds", 300)
        self.max_reasoning_steps = cfg.get("max_reasoning_steps", 5)  # Configurable steps
        self.scorer = SICQLScorer(cfg=self.cfg.get("sicql", {}), memory=memory, logger=logger)
        self.hrm_scorer = HRMScorer(cfg=self.cfg.get("hrm", {}), memory=memory, logger=logger)  # Optional use

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log("EpistemicPlanExecutorStarted", {})

        goal_dict = context.get("goal", {})
        goal_text = goal_dict.get("goal_text", "")
        goal_id = goal_dict.get("id", "unknown_goal")
        input_data = context.get("input_data", {})
        plan_to_execute = context.get("plan")

        goal_embedding = self.memory.embedding.get_or_create(goal_text)
        trace_id = f"trace_{uuid.uuid4().hex}"
        plan_signature = str(plan_to_execute)

        execution_steps: List[ExecutionStep] = []
        final_output_text: str = ""
        final_scores: Optional[ScoreBundle] = None
        final_output_embedding: Optional[List[float]] = None

        try:
            if hasattr(plan_to_execute, "trace") and callable(plan_to_execute.trace):
                trace_outputs = plan_to_execute.trace(goal=goal_dict, input_data=input_data)
            else:
                trace_outputs = [
                    f"Step 1: Analyzing goal '{goal_text[:20]}...'",
                    f"Step 2: Retrieving relevant information for '{goal_text.split()[0]}'...",
                    "Step 3: Synthesizing findings from retrieved data...",
                ]

            step_id = int(time.time() * 1000)
            for i, step_output_text in enumerate(trace_outputs):
                step_id += 1
                step_description = f"Reasoning step {i + 1}"

                try:
                    scorable_dict = {"text": step_output_text, "id": step_id}
                    scorable = ScorableFactory.from_dict(scorable_dict, TargetType.DOCUMENT)

                    sicql_scores = self.scorer.score(goal=goal_dict, scorable=scorable, dimensions=self.dimensions)
                    hrm_scores = self.hrm_scorer.score(goal=goal_dict, scorable=scorable, dimensions=self.dimensions)

                    exec_step = ExecutionStep(
                        step_id=step_id,
                        description=step_description,
                        output_text=step_output_text,
                        scores=sicql_scores,
                        output_embedding=None,
                        meta={
                            "sicql_scores": sicql_scores.to_dict(),
                            "hrm_scores": hrm_scores.to_dict()
                        },
                    )
                    execution_steps.append(exec_step)

                except Exception as e:
                    self.logger.log("EpistemicPlanExecutorStepError", {
                        "message": f"Error scoring step {step_id}.",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })
                    continue

            final_output_text = f"Final Answer: Based on the analysis of '{goal_text}', the conclusion is..."

            try:
                final_scorable_dict = {"text": final_output_text, "id": int(time.time() * 1000)}
                final_scorable = ScorableFactory.from_dict(final_scorable_dict, TargetType.DOCUMENT)
                final_scores = self.scorer.score(goal=goal_dict, scorable=final_scorable, dimensions=self.dimensions)
                final_output_embedding = self.memory.embedding.get_or_create(final_output_text)

            except Exception as e:
                self.logger.log("EpistemicPlanExecutorFinalScoringError", {
                    "message": "Error scoring final output.",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        except Exception as e:
            self.logger.log("EpistemicPlanExecutorExecutionError", {
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
            context["executed_plan_trace"] = None
            context["epistemic_executor_status"] = "failed"
            return context

        try:
            executed_trace = PlanTrace(
                trace_id=trace_id,
                goal_text=goal_text,
                goal_embedding=goal_embedding,
                input_data=input_data,
                plan_signature=plan_signature,
                execution_steps=execution_steps,
                final_output_text=final_output_text,
                final_scores=final_scores,
                final_output_embedding=final_output_embedding,
                target_epistemic_quality=None,
                target_epistemic_quality_source=None,
                created_at="",
                meta={
                    "goal_id": goal_id,
                    "executor_agent": self.__class__.__name__,
                    "plan_type": type(plan_to_execute).__name__,
                },
            )

            self.save_trace_markdown(executed_trace)

            context["executed_plan_trace"] = executed_trace
            context["epistemic_executor_status"] = "completed"
            self.logger.log("EpistemicPlanExecutorCompleted", {
                "trace_id": trace_id,
                "num_execution_steps": len(execution_steps),
            })

        except Exception as e:
            self.logger.log("EpistemicPlanExecutorAssemblyError", {
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
            context["executed_plan_trace"] = None
            context["epistemic_executor_status"] = "failed"

        return context

    def format_trace_as_markdown(self, trace: PlanTrace) -> str:
        lines = [f"## Plan Trace: {trace.trace_id}", f"**Goal:** {trace.goal_text}\n"]
        for step in trace.execution_steps:
            lines.append(f"### Step {step.step_id}: {step.description}")
            lines.append(f"Output: `{step.output_text}`")
            lines.append(step.scores.to_report(f"Step {step.step_id}: Scores"))
        lines.append(f"\n**Final Output:** `{trace.final_output_text}`")
        lines.append("Final Scores:")
        lines.append(trace.final_scores.to_report("Trace Final Scores") if trace.final_scores else "No final scores available.")
        return "\n".join(lines)
    
    import os

    def save_trace_markdown(self, trace: PlanTrace, reports_dir: str = "reports") -> str:
        """
        Saves the PlanTrace as a markdown file in the reports directory.

        Args:
            trace (PlanTrace): The trace object to convert and save.
            reports_dir (str): Directory where the report should be saved.

        Returns:
            str: The path to the saved markdown file.
        """
        os.makedirs(reports_dir, exist_ok=True)
        markdown_text = self.format_trace_as_markdown(trace)
        filename = f"{trace.trace_id}.md"
        filepath = os.path.join(reports_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        print(f"Trace saved to {filepath}")
        self.logger.log("EpistemicTraceSaved", {"path": filepath})
        return filepath
