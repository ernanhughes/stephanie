# co_ai/agents/generation.py
import re

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, HYPOTHESES


class GenerationAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")

        self.log(f"Generating hypotheses for: {goal}")

        # Load literature if available
        literature = context.get("literature", {})

        # Build context for prompt
        render_context = {
            "literature": literature,
            "feedback": context.get("feedback", {}),
            HYPOTHESES: context.get(HYPOTHESES, [])
        }

        # Load prompt based on strategy
        prompt = self.prompt_loader.load_prompt(self.cfg,
            context={**render_context, **context}
        )

        # Call LLM
        response = self.call_llm(prompt).strip()

        # Extract hypotheses
        hypotheses = self.extract_hypotheses(response)

        for h in hypotheses:
            self.memory.hypotheses.store(goal, h, 0.0, None, None)

        self.log(f"Parsed {len(hypotheses)} hypotheses.")
        
        # Update context with new hypotheses
        context[self.output_keys] = hypotheses

        # Log event
        self.logger.log("GeneratedHypotheses", {
            GOAL: goal,
            HYPOTHESES: hypotheses,
            "prompt_snippet": prompt[:300],
            "response_snippet": response[:500]
        })

        return context

    @staticmethod
    def extract_hypotheses(text: str):
        # First attempt: Try precise regex-based extraction
        pattern = re.compile(
            r"(# Hypothesis\s+\d+\s*\n(?:.*?\n)*?)(?=(# Hypothesis\s+\d+|\Z))",
            re.IGNORECASE
        )
        matches = list(pattern.finditer(text))

        if matches:
            return [match.group(1).strip() for match in matches]

        # Fallback: Split on the word "hypotheses" and rebuild sections
        split_parts = re.split(r"\bHypothesis\s+\d+\b", text, flags=re.IGNORECASE)

        if len(split_parts) <= 1:
            return []  # No valid hypotheses found at all

        # Reconstruct each hypothesis section
        hypotheses = []
        for i, part in enumerate(split_parts[1:], start=1):  # Skip preamble
            cleaned = part.strip()
            if cleaned:
                hypotheses.append(f"Hypothesis {i} {cleaned}")

        return hypotheses
