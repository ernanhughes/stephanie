
from collections import defaultdict

from co_ai.agents import BaseAgent
from co_ai.constants import GOAL, PIPELINE


class SymbolicOptimizerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, {})
        goal_type = goal.get("goal_type", "unknown")

        # Retrieve score history for this goal type
        score_history = self.memory.scores.get_scores_by_goal_type(goal_type)

        # Analyze past scores to determine best-performing pipeline
        best_pipeline = self.find_best_pipeline(score_history)

        if best_pipeline:
            context["symbolic_suggestion"] = {
                "goal_type": goal_type,
                "suggested_pipeline": best_pipeline,
                "source": "SymbolicOptimizer"
            }
            self.logger.log("SymbolicPipelineSuggestion", {
                "goal_type": goal_type,
                "suggested_pipeline": best_pipeline
            })

        return context

    def find_best_pipeline(self, score_history):
        scores_by_pipeline = defaultdict(list)

        for score in score_history:
            pipeline = tuple(score.metadata.get(PIPELINE, []))
            if pipeline and score.score is not None:
                scores_by_pipeline[pipeline].append(score.score)

        pipeline_scores = {
            pipe: sum(vals) / len(vals)
            for pipe, vals in scores_by_pipeline.items()
            if len(vals) >= 2
        }

        if not pipeline_scores:
            return None

        best = max(pipeline_scores.items(), key=lambda x: x[1])
        return list(best[0])
