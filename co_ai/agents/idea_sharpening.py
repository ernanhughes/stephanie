# co_ai/agents/idea_sharpening.py

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, HYPOTHESES, PIPELINE, PIPELINE_RUN_ID
from co_ai.evaluator import MRQSelfEvaluator
from co_ai.models import HypothesisORM


class IdeaSharpeningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.target = cfg.get("target", "generation")
        self.device = cfg.get("device", "cpu")
        self.evaluator = MRQSelfEvaluator(memory, logger, device=self.device)
        self.templates = cfg.get("templates", ["critic"])
        self.save_count = cfg.get("save_count", 3)


    async def run(self, context: dict) -> dict:
        """
        Main execution loop for IdeaSharpeningAgent.

        Takes a list of ideas, sharpens them using templates,
        judges against baseline using evaluator, and logs results.
        """
        goal = context.get(GOAL, {})
        ideas = context.get("ideas", [])

        if not ideas:
            self.logger.log("NoIdeasToSharpen", {"reason": "empty_input"})
            return context

        sharpened_results = []
        for idea in ideas:
            idea_text = idea.get("idea_text")
            result = await self._sharpen_and_evaluate(idea_text, goal, context)
            sharpened_results.append(result)

        # Sort by score
        sharpened_results.sort(key=lambda x: x["score"], reverse=True)

        # Update context
        context["sharpened_ideas"] = [r["sharpened_hypothesis"] for r in sharpened_results]
        context["scored_ideas"] = sharpened_results
        best_idea = sharpened_results[0]["sharpened_hypothesis"]
        context["top_idea"] = best_idea

        hypotheses = context.get(HYPOTHESES, [])
        if hypotheses:
            # Find the hypothesis with the maximum confidence value
            sorted_hyps = sorted(
                hypotheses, key=lambda h: h.get("confidence", 0.0), reverse=True
            )

            # Keep only the top hypothesis
            context[HYPOTHESES] = sorted_hyps[:self.save_count]
            # For scoring later
            context["baseline_hypotheses"] = sorted_hyps[-1]

        return context

    async def _sharpen_and_evaluate(self, idea: str, goal: dict, context: dict) -> dict:
        # Build prompt for refinement
        focus_area = goal.get("focus_area", "")
        baselines = self.cfg.get("baselines")
        baseline = baselines.get(focus_area, baselines.get("default"))
        merged = {
            "goal": goal,
            "idea": idea,
            "baseline": baseline,
            "literature_summary": context.get("knowledge_base_summaries", []),
            "examples": self.memory.hypotheses.get_similar(idea, limit=3),
            "strategy": goal.get("strategy", "default"),
        }

        improved = None
        winner = "original"
        scores = {}

        for name in self.templates:
            prompt_template = self.prompt_loader.from_file(name, self.cfg, merged)
            sharpened = self.call_llm(prompt_template, merged)

            try:
                preferred_output, scores = self.evaluator.judge(
                    prompt=idea,
                    output_a=idea,
                    output_b=sharpened,
                    context=merged,
                )
                improved = preferred_output
                winner = "b" if improved == sharpened else "a"
            except Exception as e:
                self.logger.log("IdeaSharpeningFailed", {"error": str(e)})
                improved = idea
                winner = "a"
                scores = {"value_a": 5.0, "value_b": 5.0}

            result = {
                "template_used": name,
                "original_idea": idea,
                "sharpened_hypothesis": improved,
                "winner": winner,
                "improved": winner == "b",
                "scores": scores,
                "score": max(scores.values()),
                "pipeline_stage": context.get(PIPELINE),
                "prompt_template": prompt_template,
            }

            saved_hyp = self.save_improved(goal, idea, result, context)
            if saved_hyp:
                context.setdefault(HYPOTHESES, []).append(saved_hyp.to_dict())
            return result

    def save_improved(self, goal: dict, original_idea: str, result: dict, context: dict):
        if not result["improved"]:
            return None
        sharpened = result["sharpened_hypothesis"]
        prompt_id = self.memory.prompt.get_id_from_response(sharpened)

        # Save to HypothesisORM
        hyp = HypothesisORM(
            goal_id=goal.get("id"),
            text=sharpened,
            prompt_id=prompt_id,
            pipeline_signature=context.get(PIPELINE),
            pipeline_run_id=context.get(PIPELINE_RUN_ID),
            source="idea_sharpening_agent",
            confidence=result["score"]
        )
        self.memory.hypotheses.insert(hyp)

        self.logger.log(
            "IdeaSharpenedAndSaved",
            {
                "prompt_snippet": original_idea[:100],
                "response_snippet": sharpened[:100],
                "score": result["score"],
            },
        )

        return hyp
