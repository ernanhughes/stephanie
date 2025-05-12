# co_ai/agents/generation.py
import re
from typing import Dict, Any

from co_ai.agents.base import BaseAgent


class GenerationAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", "")
        self.log(f"Generating hypotheses for: {goal}")

        # Load literature if available
        literature = context.get("literature", {})

        # Build context for prompt
        render_context = {
            "literature": literature,
            "feedback": context.get("feedback", {}),
            "hypotheses": context.get("hypotheses", [])
        }

        # Load prompt based on strategy
        prompt = self.prompt_loader.load_prompt(self.cfg,
            context={**render_context, **context}
        )

        # Call LLM
        response = self.call_llm(prompt).strip()

        # Extract hypotheses
        hypotheses = self.extract_hypothesis(response)

        for h in hypotheses:
            self.memory.store_hypothesis(goal, h, 0.0, None, None)

        self.log(f"Parsed {len(hypotheses)} hypotheses.")
        
        # Update context with new hypotheses
        context[self.output_keys] = hypotheses

        # Log event
        self.logger.log("GeneratedHypotheses", {
            "goal": goal,
            "hypotheses": hypotheses,
            "prompt_snippet": prompt[:300],
            "response_snippet": response[:500]
        })

        return context

    import re

    def extract_hypothesis(self,  text: str) -> list[str]:
        pattern = re.compile(
            r"(Hypothesis\s+\d+:\s*[\s\S]*?)(?=Hypothesis\s+\d+:|\Z)",  # grab from "Hypothesis X:" to next or end
            re.IGNORECASE
        )
        return [match.strip() for match in pattern.findall(text)]
