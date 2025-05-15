# co_ai/agents/generation.py
import re

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, HYPOTHESES, LITERATURE, FEEDBACK, NAME


class GenerationAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")

        self.logger.log("GenerationStart", {GOAL: goal})

        # Load literature if available
        literature = context.get(LITERATURE, {})

        # Build context for prompt
        render_context = {
            LITERATURE: literature,
            FEEDBACK: context.get(FEEDBACK, {}),
            HYPOTHESES: context.get(HYPOTHESES, []),
        }

        # Load prompt based on strategy
        prompt = self.prompt_loader.load_prompt(
            self.cfg, context={**render_context, **context}
        )
        context[f"{self.cfg.get(NAME)}_prompt"] = prompt

        response = self.call_llm(prompt)

        # Extract hypotheses
        hypotheses = self.extract_hypotheses(response)

        for h in hypotheses:
            self.memory.hypotheses.store(goal, h, 0.0, None, None, prompt)

        merged = {**context, **{"input_prompt":prompt, "example_output":response}}

        prompt_improved_prompt = self.prompt_loader.from_file("improve.txt", self.cfg, merged)

        # Call LLM
        improved_prompt = self.call_llm(prompt_improved_prompt)

        improved_response = self.call_llm(improved_prompt)


        hypotheses = self.extract_hypotheses(improved_response)

        for h in hypotheses:
            self.memory.hypotheses.store(goal, h, 0.0, None, None, improved_prompt)


        # Update context with new hypotheses
        context[self.output_key] = hypotheses

        # Log event
        self.logger.log(
            "GeneratedHypotheses",
            {
                GOAL: goal,
                HYPOTHESES: hypotheses,
                "prompt_snippet": prompt[:300],
                "response_snippet": response[:500],
            },
        )

        return context

    @staticmethod
    def extract_hypotheses(text: str):
        # First attempt: Try precise regex-based extraction
        pattern = re.compile(
            r"(# Hypothesis\s+\d+\s*\n(?:.*?))(?:\n(?=# Hypothesis\s+\d+)|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        matches = list(pattern.finditer(text))

        if matches:
            return [match.group(1).strip() for match in matches]

        # Fallback (if needed)
        split_parts = re.split(r"\bHypothesis\s+\d+\b", text, flags=re.IGNORECASE)
        if len(split_parts) <= 1:
            return [text]

        hypotheses = []
        for i, part in enumerate(split_parts[1:], start=1):
            cleaned = part.strip()
            if cleaned:
                hypotheses.append(f"Hypothesis {i} {cleaned}")

        return hypotheses
