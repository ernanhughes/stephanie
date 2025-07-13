# stephanie/agents/compiler/step_compiler.py

from dataclasses import asdict

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.memory_aware_mixin import MemoryAwareMixin
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.rules.symbolic_node import SymbolicNode
from stephanie.scoring.mrq.mrq_scorer import MRQScorer


class StepCompilerAgent(ScoringMixin, MemoryAwareMixin, BaseAgent):
    """
    Breaks down a high-level goal into symbolic reasoning steps.
    Each step is a SymbolicNode with step_id, action, description, etc.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.scorer = MRQScorer(cfg=cfg, memory=memory, logger=logger)
        self.scorer.load_models()

    async def run(self, context: dict) -> dict:
        goal = context.get("goal")

        # Augment context with shared memory
        context = self.inject_memory_context(
            goal=goal, context=context, tags=["step", "plan"]
        )
        prompt = self.prompt_loader.load_prompt(self.cfg, context=context)

        # Generate response from LLM
        response = self.call_llm(prompt, context=context)
        steps = self.parse_response_into_steps(response)
        context["step_plan"] = steps

        # Score the plan using MRQ
        try:
            score_result = self.score_item(
                {"text": response},
                context,
                metrics="step_reasoning",
                scorer=self.scorer,
            )
            context["step_plan_score"] = score_result.aggregate()
            context.setdefault("dimension_scores", {})["step_plan"] = (
                score_result.to_dict()
            )
        except Exception as e:
            self.logger.log("StepCompilerScoringError", {"error": str(e)})
            context["step_plan_score"] = None

        # Store trace in shared memory
        self.add_to_shared_memory(
            context,
            {
                "agent": "step_compiler",
                "trace": "\n".join([s["description"] for s in steps]),
                "response": response,
                "score": context.get("step_plan_score"),
                "dimension_scores": context.get("dimension_scores", {}).get(
                    "step_plan"
                ),
                "tags": ["step", "plan"],
            },
        )

        return context

    def parse_response_into_steps(self, response: str):
        """
        Parses a multi-line LLM response into a list of symbolic steps.
        Each step becomes a SymbolicNode (or plain dictionary if symbolic layer is deferred).
        """
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        steps = []
        for i, line in enumerate(lines):
            if ":" in line:
                _, description = line.split(":", 1)
                step = SymbolicNode(
                    step_name=f"step_{i + 1}",
                    action="reasoning_step",
                    description=description.strip(),
                    metadata={"source": "step_compiler"},
                )
                steps.append(asdict(step))
        return steps
