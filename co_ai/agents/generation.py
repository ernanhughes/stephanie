# co_ai/agents/generation.py

from co_ai.agents.base_agent import BaseAgent
from co_ai.constants import (FEEDBACK, GOAL, GOAL_TEXT, HYPOTHESES, LITERATURE,
                             PIPELINE, PIPELINE_RUN_ID)
from co_ai.parsers import extract_hypotheses
from co_ai.tools.huggingface_tool import recommend_similar_papers


class GenerationAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)

        self.logger.log("GenerationStart", {GOAL: goal})

        # Load literature if available
        literature = context.get(LITERATURE, {})

        # Build context for prompt
        render_context = {
            GOAL: goal.get(GOAL_TEXT),
            LITERATURE: literature,
            FEEDBACK: context.get(FEEDBACK, {}),
            HYPOTHESES: context.get(HYPOTHESES, []),
        }
        merged = {**context, **render_context}

        # Load prompt based on strategy
        prompt_text = self.prompt_loader.load_prompt(self.cfg, merged)
        response = self.call_llm(prompt_text, context)

        # Extract hypotheses
        hypotheses = extract_hypotheses(response)
        hypotheses_saved = []
        prompt = self.memory.prompt.get_from_text(prompt_text)
        for h in hypotheses:
            hyp = self.save_hypothesis(
                {
                    "text": h,
                    "prompt_id": prompt.id,
                },
                context=context,
            )
            hypotheses_saved.append(hyp.to_dict())

        # Update context with new hypotheses
        context[self.output_key] = hypotheses_saved

        # Log event
        self.logger.log(
            "GeneratedHypotheses",
            {
                GOAL: goal,
                HYPOTHESES: hypotheses,
                "prompt_snippet": prompt_text[:100],
                "response_snippet": response[:200],
            },
        )

        return context
