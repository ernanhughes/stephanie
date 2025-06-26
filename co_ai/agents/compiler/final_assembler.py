from co_ai.agents.base_agent import BaseAgent
from co_ai.constants import GOAL
from co_ai.scoring.mrq_scorer import MRQScorer
from co_ai.models.tokenizer import TokenCounter
from co_ai.utils.llm import call_llm
from co_ai.utils.memory_manager import SharedMemoryManager


class FinalAssemblerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.token_limit = cfg.get("token_limit", 2048)
        self.scorer = MRQScorer(cfg, memory, logger)
        self.token_counter = TokenCounter(model_name="qwen:7b")
        self.memory_manager = SharedMemoryManager(memory=memory)
        self.max_steps = cfg.get("max_steps", 15)  # Limit on how many steps to include

    async def run(self, context: dict) -> dict:
        goal = self.extract_goal_text(context.get(GOAL))
        steps = context.get("step_outputs", [])
        if not steps:
            self.logger.log("FinalAssemblySkipped", {"reason": "no_steps"})
            return context

        self.logger.log("FinalAssemblyStart", {
            "step_count": len(steps),
            "goal_snippet": goal[:100]
        })

        # 1. Score & Sort Steps (by quality + relevance)
        scored_steps = self._score_and_sort_steps(steps, goal)

        # 2. Assemble Prompt with Diversity & Coverage
        final_prompt, used_indices = self._assemble_prompt(scored_steps)

        # 3. Optional: Evaluate final prompt via LLM + MR.Q
        final_score = None
        try:
            output = call_llm(final_prompt, context={"goal": goal})
            score_obj = self.scorer.score(hypothesis={"text": output}, context={"goal": goal})
            final_score = score_obj.aggregate()
        except Exception as e:
            self.logger.log("FinalPromptScoringFailed", {"error": str(e)})

        # 4. Log and store result
        self.memory_manager.save_final_prompt(
            goal=goal,
            prompt=final_prompt,
            score=final_score,
            used_indices=used_indices,
            steps_used=[steps[i] for i in used_indices]
        )

        # 5. Update context
        context["final_prompt"] = final_prompt
        context["final_score"] = final_score
        context["final_assembly_steps"] = [
            {
                "step_index": idx,
                "snippet": steps[idx]["step"][:100],
                "score": steps[idx].get("score", 0),
            }
            for idx in used_indices
        ]

        self.logger.log("FinalAssemblyComplete", {
            "final_score": final_score,
            "assembled_prompt_snippet": final_prompt[:200],
            "token_count": self.token_counter.count_tokens(final_prompt),
        })

        return context

    def _score_and_sort_steps(self, steps, goal):
        """
        Enhance each step with relevance score based on goal.
        Sort by combined score (original score + goal alignment).
        """
        scored_steps = []
        for i, step in enumerate(steps):
            step_text = step.get("step", "")
            original_score = step.get("score", 0)

            # Estimate goal alignment (using MR.Q-style embedding match)
            goal_alignment = self.scorer.predict_score_from_prompt(
                prompt=f"Step: {step_text}\nGoal: {goal}",
                dimension="relevance"
            )

            combined_score = (original_score * 0.6) + (goal_alignment * 0.4)

            scored_steps.append({
                "index": i,
                "step": step_text,
                "original_score": original_score,
                "goal_alignment": goal_alignment,
                "combined_score": combined_score
            })

        # Sort by combined score descending
        scored_steps.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_steps

    def _assemble_prompt(self, scored_steps):
        """
        Assemble steps into a single prompt while ensuring:
        - Token limit is respected
        - Content is diverse and covers multiple angles
        - High-quality steps are prioritized
        """
        final_prompt = ""
        used_indices = []

        for item in scored_steps:
            index = item["index"]
            step_text = item["step"]

            # Skip empty or duplicate steps
            if not step_text.strip() or step_text in final_prompt:
                continue

            candidate_prompt = final_prompt + "\n\n" + step_text
            tokens = self.token_counter.count_tokens(candidate_prompt)

            if tokens > self.token_limit:
                self.logger.log("FinalAssemblyTokenLimitReached", {
                    "step_index": index,
                    "tokens": tokens,
                    "step_snippet": step_text[:100]
                })
                break

            final_prompt = candidate_prompt.strip()
            used_indices.append(index)

            # Stop after max_steps
            if len(used_indices) >= self.max_steps:
                break

        return final_prompt, used_indices