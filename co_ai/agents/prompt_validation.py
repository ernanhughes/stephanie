from collections import defaultdict

from co_ai.agents.base import BaseAgent
from co_ai.constants import PIPELINE_RUN_ID


class PromptValidationAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        self.logger.log("PromptValidationAgentStart", {PIPELINE_RUN_ID: context.get(PIPELINE_RUN_ID)})

        hypotheses = context.get("scored_hypotheses", [])
        compiled_prompt = context.get("compiled_prompt", {})
        compiled_prompt_hash = hash(compiled_prompt.get("text", ""))

        prompt_to_scores = defaultdict(list)

        for hypo in hypotheses:
            prompt_key = hypo.get("source_prompt")
            score = hypo.get("score")
            if prompt_key is not None and score is not None:
                prompt_to_scores[prompt_key].append(score)

        if not prompt_to_scores:
            self.logger.log("PromptValidationNoData", {"message": "No prompt-score mappings found."})
            return context

        prompt_avg_scores = {
            prompt: sum(scores) / len(scores)
            for prompt, scores in prompt_to_scores.items()
        }

        best_prompt, best_score = max(prompt_avg_scores.items(), key=lambda x: x[1])

        passed_validation = compiled_prompt_hash == best_prompt
        self.logger.log("PromptValidationResult", {
            "compiled_prompt_hash": compiled_prompt_hash,
            "best_prompt_hash": best_prompt,
            "best_score": best_score,
            "passed": passed_validation,
        })

        print("\n=== Prompt Validation ===")
        print(f"Selected prompt hash: {compiled_prompt_hash}")
        print(f"Best-scoring prompt hash: {best_prompt} (avg score: {best_score:.2f})")
        if passed_validation:
            print("\u2705 Prompt compiler selected the best-performing prompt!")
        else:
            print("\u26a0\ufe0f Mismatch: Consider revising compiler strategy or scoring alignment.")

        return context
