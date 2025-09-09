# stephanie/agents/generation.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import (DOCUMENTS, FEEDBACK, GOAL, GOAL_TEXT,
                                 HYPOTHESES, SCORABLES)
from stephanie.utils.parser_utils import extract_hypotheses_with_score


class GenerationAgent(BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)

        self.logger.log("GenerationStart", {GOAL: goal})

        # --- Load documents (scorables) ---
        scorables = context.get(DOCUMENTS, []) or []
        
        # Sort by relevance (if available), else just truncate
        # Assumes each scorable has 'score' and 'summary'
        sorted_scorables = sorted(
            scorables,
            key=lambda s: s.get("score", 0),
            reverse=True
        )

        # Select top-5
        top_scorables = sorted_scorables[:5]

        # Replace text with summary if available
        for s in top_scorables:
            if "summary" in s and s["summary"]:
                s["text"] = s["summary"]

        # Build context for prompt
        render_context = {
            GOAL: goal.get(GOAL_TEXT),
            SCORABLES: top_scorables,
            FEEDBACK: context.get(FEEDBACK, {}),
            HYPOTHESES: context.get(HYPOTHESES, []),
        }
        merged = {**context, **render_context}

        # Load prompt based on strategy
        prompt_text = self.prompt_loader.load_prompt(self.cfg, merged)
        response = self.call_llm(prompt_text, context)

        # --- Hypothesis extraction remains the same ---
        hypotheses = extract_hypotheses_with_score(response)
        hypotheses_saved = []
        prompt = self.memory.prompts.get_from_text(prompt_text)
        for h in hypotheses:
            hyp = self.save_hypothesis(
                {
                    "text": h["text"],
                    "metadata": {
                        "rationale": h.get("rationale"),
                        "score": h.get("score"),
                        "header": h.get("header"),
                        "prompt_id": prompt.id if prompt else None,
                    },
                },
                context=context,
            )
            hypotheses_saved.append(hyp.to_dict())

        self.set_scorable_details(
            input_text=prompt_text,
            output_text="\n\n".join(hyp["text"] for hyp in hypotheses_saved),
            description=f"Hypotheses for goal: {goal.get(GOAL_TEXT, '')}",
        )

        # Update context
        context[self.output_key] = hypotheses_saved

        self.report(
            {
                "event": "GeneratedHypotheses",
                "step": "Generation",
                "details": [h["text"] for h in hypotheses_saved[:2]],
            }
        )
        return context
