from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, HYPOTHESES, PIPELINE, PIPELINE_RUN_ID
from stephanie.parsers import extract_hypotheses


class RefinerAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        goal = self.extract_goal_text(context.get(GOAL))
        target_agent = self.cfg.get("target_agent", "generation")
        history = context.get("prompt_history", {}).get(target_agent, None)

        if not history:
            self.logger.log("RefinerNoHistoryFound", {
                "target_agent": target_agent,
                "context_keys": list(context.keys())
            })
            return context

        self.logger.log("RefinerStart", {
            "target_agent": target_agent,
            "goal": goal
        })

        original_prompt = history["prompt"]
        original_response = history["response"]
        preferences = history.get("preferences", [])
        original_hypotheses = context.get(HYPOTHESES, [])

        merged = {
            **context,
            "input_prompt": original_prompt,
            "example_output": original_response,
            "preferences": preferences,
        }

        try:
            prompt_improved_prompt = self.prompt_loader.load_prompt(
                self.cfg, context=merged
            )
            self.logger.log("RefinerImprovementPromptLoaded", {
                "snippet": prompt_improved_prompt[:200]
            })

            refined_prompt = self.call_llm(prompt_improved_prompt, context)
            self.logger.log("RefinerPromptGenerated", {
                "prompt_snippet": refined_prompt[:200]
            })

            refined_response = self.call_llm(refined_prompt, context)
            self.logger.log("RefinerResponseGenerated", {
                "response_snippet": refined_response[:200]
            })

            refined_hypotheses = extract_hypotheses(refined_response)
            self.logger.log(
                "RefinerHypothesesExtracted", {"count": len(refined_hypotheses)}
            )

            refined = []
            for h in refined_hypotheses:
                hyp = self.save_hypothesis(
                    {
                        "goal": goal,
                        "text": h,
                        "prompt": refined_prompt,
                        "pipeline_run_id": context.get(PIPELINE_RUN_ID),
                        "pipeline_signature": context.get(PIPELINE),
                    },
                    context=context
                )
                refined.append(hyp)

            info = {
                "original_response": original_response,
                "original_hypotheses": original_hypotheses,
                "refined_prompt": refined_prompt,
                "refined_hypotheses": refined_hypotheses
            }
            refined_merged = {**merged, **info}

            evaluation_template = self.cfg.get("evaluation_template", "evaluate.txt")
            evaluation_prompt = self.prompt_loader.from_file(
                evaluation_template, self.cfg, refined_merged
            )
            self.logger.log("RefinerEvaluationPromptGenerated", {
                "snippet": evaluation_prompt[:200]
            })

            evaluation_response = self.call_llm(evaluation_prompt, context)
            self.logger.log("RefinerEvaluationResponse", {
                "snippet": evaluation_response[:200]
            })

            if " 2" in evaluation_response:
                context[HYPOTHESES] = refined_hypotheses
                self.logger.log("RefinedUpdated", info)
            else:
                self.logger.log("RefinedSkipped", info)

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log("RefinerError", {
                "error": str(e),
                "context_keys": list(context.keys())
            })

        return context
