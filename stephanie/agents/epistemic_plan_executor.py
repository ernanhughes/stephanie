# stephanie/agents/epistemic_plan_executor.py

import json
import re
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

import dspy

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.plan_trace import ExecutionStep, PlanTrace
from stephanie.scoring.hrm_scorer import HRMScorer  # Optional
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.sicql_scorer import SICQLScorer


# Define a Signature for a single LATS-style reasoning step
class ReasoningStepSignature(dspy.Signature):
    """Generate the next logical reasoning step towards solving a goal."""
    
    goal = dspy.InputField(desc="The main goal to solve.")
    previous_steps_summary = dspy.InputField(desc="A concise summary of the previous reasoning steps taken so far.")
    input_data = dspy.InputField(desc="Any initial data or context provided for the task.", format=lambda x: json.dumps(x, indent=2))
    step_number = dspy.InputField(desc="The current step number in the sequence.")
    
    # The output field instructs the model on the expected format
    next_step = dspy.OutputField(desc="The next reasoning step. Be specific and build on prior steps. "
                                      "If you have logically concluded the task, start your response EXACTLY with 'Final Answer: ' followed by your conclusion.")

FINAL_ANSWER_PATTERN = re.compile(r"(?:^|\n)\s*final\s*answer\s*[:：]\s*", re.IGNORECASE)


class EpistemicPlanExecutorAgent(BaseAgent):
    """
    Agent to execute a reasoning plan using a simplified, internal LATS-like process
    and generate a detailed PlanTrace for subsequent analysis by the Epistemic Plan HRM.
    This avoids direct dependency on the external LATSDSPyAgent.
    """

    def __init__(
        self, cfg: Dict[str, Any], memory: Any = None, logger: Any = None
    ):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", [])
        self.plan_timeout_seconds = cfg.get("plan_timeout_seconds", 300)
        self.max_reasoning_steps = cfg.get("max_reasoning_steps", 5) # Configurable steps
        self.use_hrm_in_trace = cfg.get("use_hrm_in_trace", True) # Config flag

        self.sicql_scorer = SICQLScorer(cfg=self.cfg.get("sicql", {}), memory=memory, logger=logger)
        if self.use_hrm_in_trace:
            self.hrm_scorer = HRMScorer(cfg=self.cfg.get("hrm", {}), memory=memory, logger=logger)
        else:
            self.hrm_scorer = None
        # Get the configured LM
        self.lm = dspy.LM(
            "ollama_chat/qwen3",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=self.lm)
        self.step_predictor = dspy.ChainOfThought(
            signature=ReasoningStepSignature
        )
        self.logger.log("EpistemicPlanExecutorAgentInitialized", {
            "max_reasoning_steps": self.max_reasoning_steps,
            "use_hrm_in_trace": self.use_hrm_in_trace,
        })

    async def _run_simplified_lats(self, goal_text: str, input_data: Dict[str, Any]) -> List[str]:
        """
        Simplified internal logic to generate a sequence of reasoning steps,
        using dspy.Predict/ChainOfThought for structured prompting.

        Args:
            goal_text (str): The main goal to reason about.
            input_data (dict): Initial data provided to the reasoning process.

        Returns:
            List[str]: A list of strings, each representing an intermediate reasoning step/output.
        """
        trace_outputs = []
        # Start with an empty summary; the predictor can handle this.
        previous_steps_summary = "" 

        for step_num in range(1, self.max_reasoning_steps + 1):
            # self.logger.log("LATS_StepStarted", {"step": step_num, "summary": previous_steps_summary[-100:]})
            self.logger.log("LATS_StepStarted", {"step": step_num})

            try:
                # --- Use dspy.Predict/ChainOfThought to generate the next step ---
                # Prepare the prediction inputs based on the Signature
                prediction_kwargs = {
                    "goal": goal_text,
                    "previous_steps_summary": previous_steps_summary,
                    "input_data": input_data,
                    "step_number": step_num
                }
                
                prediction = self.step_predictor(**prediction_kwargs)
                
                # --- Extract the Output ---
                # The output is accessed via the attribute name defined in the Signature ('next_step')
                step_output_text = prediction.next_step.strip()

                # --- Check for Final Answer ---
                is_final_answer = bool(FINAL_ANSWER_PATTERN.search(step_output_text))

                is_final_answer = step_output_text.lower().startswith("final answer: ")
                if is_final_answer:
                    # Extract the part after "Final Answer: "
                    # final_part = step_output_text[len("final answer: "):].strip()
                    # trace_outputs.append(f"Final Answer: {final_part}")
                    # Let's keep the full text including the prefix for clarity in the trace
                    trace_outputs.append(step_output_text) 
                    self.logger.log("EpistemicPlanExecutorLATS", {
                        "message": f"Early stopping at step {step_num} due to 'Final Answer' signal.",
                        "final_answer_snippet": step_output_text[:100]
                    })
                    break # Stop the loop
                else:
                    trace_outputs.append(step_output_text)
                    # Update the summary for the next step
                    # A more robust summary could be built, but for now, append the last step
                    # Truncate previous summary and current step to keep it manageable
                    if len(previous_steps_summary) > 300:
                        previous_steps_summary = previous_steps_summary[-200:]
                    previous_steps_summary += f"\nStep {step_num}: {step_output_text[:100]}..."
                    # Ensure it doesn't grow too large
                    if len(previous_steps_summary) > 500:
                        previous_steps_summary = previous_steps_summary[-400:]

                self.logger.log("LATS_StepCompleted", {"step": step_num, "output_snippet": step_output_text[:100]})

            except Exception as e:
                self.logger.log("EpistemicPlanExecutorLATSStepError", {
                    "message": f"Error generating LATS-like step {step_num}.",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                # Decide whether to break or continue with a placeholder/error step
                trace_outputs.append(f"[ERROR: Failed to generate step {step_num}]")
                # Continue to next step

        return trace_outputs

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:

        existing_goal_ids = {
            pt.goal_id for pt in self.memory.plan_traces.all()
            if pt.goal_id is not None
        }
        goals = self.memory.goals.get_all_goals()

        for goal in goals:
            goal_id = goal.id
            # if goal.id in existing_goal_ids:
            #     self.logger.log("EpistemicPlanExecutorSkipped", {
            #         "goal_id": goal.id,
            #         "message": "Goal already has a PlanTrace, skipping."
            #     })
            #     continue

            goal_dict = goal.to_dict()
            goal_text = goal.goal_text
            if not goal_text or len(goal_text) < 10:
                self.logger.log("EpistemicPlanExecutorWarning", {
                    "message": f"Goal text is too short or missing: {goal_text}",
                    "goal_id": goal.id
                })
                continue
            
            input_data = context.get("input_data", {})
            self.logger.log("EpistemicPlanExecutorStarted", {
                "goal_id": goal_id,
                "goal_text": goal_text,
                "input_data": input_data
            })

            if not goal_text:
                error_msg = "Missing 'goal_text' in context['goal']. Cannot execute plan."
                self.logger.log("EpistemicPlanExecutorError", {"message": error_msg})
                context[self.output_key] = {
                    "goal_id": goal_id,
                    "executor_agent": self.__class__.__name__,
                    "source": "simplified_lats_execution",
                    "status": "failed",
                    "error": error_msg
                }
                return context

            trace_id = f"trace_{uuid.uuid4().hex}"
            plan_signature = f"SimplifiedLATS_{self.max_reasoning_steps}_steps"

            execution_steps: List[ExecutionStep] = []
            final_output_text: str = ""
            final_scores: Optional[ScoreBundle] = None

            try:
                # --- Execute the Simplified LATS-like Reasoning ---
                trace_outputs = await self._run_simplified_lats(goal_text, input_data)

                # --- Process Generated Trace into ExecutionSteps ---
                step_id_counter = int(time.time() * 1000)
                processed_trace_info = []

                for i, step_output_text in enumerate(trace_outputs):
                    step_id_counter += 1
                    step_description = f"Simplified LATS Step {i + 1}"
                    processed_trace_info.append({
                        "step_id": step_id_counter,
                        "description": step_description,
                        "output_text": step_output_text.strip() # Clean up whitespace
                    })

                # --- Score Each Processed Step Using Stephanie Scorers ---
                for step_info in processed_trace_info:
                    step_id = step_info["step_id"]
                    step_description = step_info["description"]
                    step_output_text = step_info["output_text"]

                    if not step_output_text:
                        self.logger.log("EpistemicPlanExecutorWarning", {
                            "message": f"Generated step {step_id} has empty output. Skipping scoring."
                        })
                        continue

                    try:
                        scorable_dict = {"text": step_output_text, "id": str(step_id)} # Ensure ID is string
                        scorable = ScorableFactory.from_dict(scorable_dict, TargetType.DOCUMENT)

                        # --- Score the Step Output ---
                        sicql_scores: ScoreBundle = self.sicql_scorer.score(
                            goal=goal_dict, scorable=scorable, dimensions=self.dimensions
                        )
                        hrm_scores: Optional[ScoreBundle] = None
                        if self.hrm_scorer:
                            hrm_scores = self.hrm_scorer.score(
                                goal=goal_dict, scorable=scorable, dimensions=self.dimensions
                            )
                            if hrm_scores:
                                sicql_scores = sicql_scores.merge(hrm_scores)


                        # --- Create ExecutionStep Object ---
                        step_meta = {
                            "sicql_scores": sicql_scores.to_dict(),
                            "source": "simplified_lats_step"
                        }
                        if hrm_scores:
                            step_meta["hrm_scores"] = hrm_scores.to_dict()

                        exec_step = ExecutionStep(
                            step_id=str(step_id), # Ensure ID is string
                            step_type="reasoning",  # Assuming all steps are actions
                            description=step_description,
                            output_text=step_output_text,
                            scores=sicql_scores, # Primary scores for the trace
                            extra_data=step_meta,
                        )
                        execution_steps.append(exec_step)

                    except Exception as e:
                        self.logger.log("EpistemicPlanExecutorStepError", {
                            "message": f"Error scoring generated step {step_id}.",
                            "step_output_snippet": step_output_text[:50],
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        })
                        continue # Continue with other steps

                # --- Determine Final Output ---
                # The final output is typically the last step's text
                # Or, if the last step started with "Final Answer:", extract that part
                if execution_steps:
                    last_step_text = execution_steps[-1].output_text
                    if last_step_text.lower().startswith("final answer:"):
                        # Extract the part after "Final Answer:"
                        final_output_text = last_step_text[len("final answer:"):].strip()
                    else:
                        final_output_text = last_step_text
                else:
                    final_output_text = "No reasoning steps were generated."

                # --- Score the Final Output ---
                try:
                    final_scorable_dict = {"text": final_output_text, "id": f"{trace_id}_final"}
                    final_scorable = ScorableFactory.from_dict(final_scorable_dict, TargetType.DOCUMENT)
                    final_scores: ScoreBundle = self.sicql_scorer.score(
                        goal=goal_dict, scorable=final_scorable, dimensions=self.dimensions
                    )

                except Exception as e:
                    self.logger.log("EpistemicPlanExecutorFinalScoringError", {
                        "message": "Error scoring final output.",
                        "final_output_snippet": final_output_text[:50],
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })

            except Exception as e:
                self.logger.log("EpistemicPlanExecutorExecutionError", {
                    "message": "Error during simplified LATS execution or trace processing.",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                context["executed_plan_trace"] = None
                context["epistemic_executor_status"] = "failed"
                context["epistemic_executor_error"] = str(e)

            # --- Assemble the PlanTrace ---
            try:
                executed_trace = PlanTrace(
                    trace_id=trace_id,
                    plan_signature=plan_signature,
                    goal_text=goal_text,
                    goal_id=goal_id,
                    input_data=input_data,
                    execution_steps=execution_steps,
                    final_output_text=final_output_text,
                    final_scores=final_scores,
                    target_epistemic_quality=final_scores.aggregate(), # To be filled later
                    target_epistemic_quality_source=self.sicql_scorer.model_type,
                    created_at="", # Can be set to current timestamp
                    extra_data={
                        "goal_id": goal_id, 
                        "executor_agent": self.__class__.__name__,
                        "source": "simplified_lats_execution",
                        "max_reasoning_steps_config": self.max_reasoning_steps
                    },
                )


                # --- Save Trace Report --- Yeah you'll be back here 1000 times OK this is going to be
                executed_trace.save_as_json(f"reports/{self.name}/")

                executed_trace.save_as_markdown(reports_dir="reports")

                # --- Store the PlanTrace and ExecutionSteps in Memory ---
                plan_trace_id = self.memory.plan_traces.add(executed_trace)
                for i, step in enumerate(execution_steps):
                    step.plan_trace_id = plan_trace_id
                    step.step_order = i + 1
                    self.memory.execution_steps.add(step)
                self.memory.session.commit()  # Commit all changes

                # --- Update Context ---
                context["executed_plan_trace"] = executed_trace
                context["epistemic_executor_status"] = "completed"
                context["epistemic_executor_error"] = None
                self.logger.log("EpistemicPlanExecutorCompleted", {
                    "trace_id": trace_id,
                    "num_execution_steps": len(execution_steps),
                    "final_output_snippet": final_output_text[:50]
                })

            except Exception as e:
                self.logger.log("EpistemicPlanExecutorAssemblyError", {
                    "message": "Error assembling PlanTrace object.",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                context[self.output_key] = {
                    "goal_id": goal_id,
                    "executor_agent": self.__class__.__name__,
                    "source": "simplified_lats_execution",
                    "max_reasoning_steps_config": self.max_reasoning_steps
                }

        return context

