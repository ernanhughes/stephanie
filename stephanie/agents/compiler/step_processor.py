# stephanie/agents/compiler/step_processor.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.memory_aware_mixin import MemoryAwareMixin
from stephanie.scoring.scorable import Scorable, ScorableType


class StepProcessorAgent(MemoryAwareMixin, BaseAgent):
    """
    Executes each reasoning step (SymbolicNode) from StepCompilerAgent,
    producing outputs and optionally scoring them.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        goal = context["goal"]
        steps = context.get(self.input_key, [])  # input_key: step_plan
        step_outputs = []

        for i, step in enumerate(steps):
            step_context = {
                "goal": goal,
                "step": step["description"],
                "step_index": i,
                **context,  # Include all existing context
            }
            prompt = self.prompt_loader.load_prompt(self.cfg, context=step_context)
            output = self.call_llm(prompt, context=step_context)

            # Score (optional)
            scorable = Scorable(text=output, type=ScorableType.HYPOTHESIS)
            score_result = self._score(
                scorable=scorable, context=context
            )
            total_score = score_result.aggregate()

            step_outputs.append(
                {
                    "step": step["description"],
                    "output": output,
                    "score": total_score,
                    "dimension_scores": score_result.to_dict(),
                }
            )

            self.logger.log(
                "StepProcessed",
                {"step": step["description"], "output": output, "score": total_score},
            )

        context["step_outputs"] = step_outputs
        return context
