import re

from co_ai.agents.base import BaseAgent

class PromptRefinerAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        # Get inputs
        goal = context.get("goal", "")
        input_prompt = context.get("current_prompt", "")
        example_output = context.get("example_output", "")
        preferences = context.get("preferences", ["goal_consistency", "simplicity"])

        # Build refinement prompt
        refinement_prompt = self._build_refinement_prompt(
            goal=goal,
            input_prompt=input_prompt,
            example_output=example_output,
            preferences=preferences
        )

        # Call LLM to get improved prompt
        try:
            response = self.call_llm(refinement_prompt)
            match = re.search(r"Refined Prompt:\n(.+)", response, re.DOTALL)
            refined_prompt = match.group(1).strip() if match else input_prompt

            # Store only if improvement detected
            if refined_prompt != input_prompt:
                self.memory.store_prompt_version(
                    agent_name=context["agent"],
                    prompt_key=context["prompt_key"],
                    prompt_text=refined_prompt,
                    source="manual_to_refined",
                    version=context["version"] + 1,
                    metadata={"improvement_score": self._evaluate_improvement(input_prompt, refined_prompt)}
                )
                context["refined_prompt"] = refined_prompt
        except Exception as e:
            self.logger.log("PromptRefiningFailed", {"error": str(e)})
            refined_prompt = input_prompt

        return context

    def _build_refinement_prompt(self, goal, input_prompt, example_output, preferences):
        """Build prompt using template + context"""
        preferences_str = ", ".join(preferences)

        return f"""
You are an expert in prompt engineering and scientific reasoning.
Your task is to refine the following prompt to produce better hypotheses.

Goal: {goal}
Preferences: {preferences_str}

Old Prompt:
{input_prompt}

Example Output:
{example_output}

Instructions:
1. Analyze the old prompt's structure and effectiveness.
2. Rewrite it to align more closely with the stated goals and preferences.
3. Ensure clarity, logical flow, and domain-specific grounding.
4. Make sure it still produces structured output like:
# Hypothesis 1
<hypothesis here>
# Hypothesis 2
<hypothesis here>
# Hypothesis 3
<hypothesis here>

5. Avoid hallucinations — keep it factual and testable.
6. Return only the refined prompt — no extra explanation.

Refined Prompt:
"""