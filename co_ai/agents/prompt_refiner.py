from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, HYPOTHESES
from co_ai.parsers import extract_hypotheses

class PromptRefinerAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        # Get inputs
        goal = context.get(GOAL, "")
        target_agent = self.cfg.get("target_agent", "generation")
        history = context.get("prompt_history", {}).get(target_agent, None)
        if history:
            original_prompt = history["prompt"]
            original_response = history["response"]
            preferences = history["preferences"]
            merged = {
                **context,
                **{
                    "input_prompt": original_prompt,
                    "example_output": original_response,
                    "preferences": preferences,
                },
            }
            original_hypotheses = context[HYPOTHESES]
            prompt_improved_prompt = self.prompt_loader.load_prompt(
                self.cfg, context=merged
            )
            # Call LLM to get the improved prompt
            refined_prompt = self.call_llm(prompt_improved_prompt, context)

            # Use the improved prompt to get the improved response
            refined_response = self.call_llm(refined_prompt, context)

            refined_hypotheses = extract_hypotheses(refined_response)

            for h in refined_hypotheses:
                self.memory.hypotheses.store(goal, h, 0.0, None, None, refined_prompt)
        
            info = {"original_response": original_response, 
                                           "original_hypotheses": original_hypotheses,
                                           "refined_prompt": refined_prompt,
                                           "refined_hypotheses": refined_hypotheses
                                           }
            refined_merged = {**merged, **info}

            evaluation_template = self.cfg.get("evaluation_template", "evaluate.txt")
            # Store the refined prompt and response in the context
            evaluation_prompt = self.prompt_loader.from_file(evaluation_template, self.cfg, refined_merged)
            evaluation_response = self.call_llm(evaluation_prompt, context)

            if evaluation_response.find(" 2"):
                context[HYPOTHESES] = refined_hypotheses
                self.log("RefinedSuccess", info)
            else:
                self.log("RefinedFailure", info)

        return context
