from dataclasses import asdict
from datetime import datetime

from co_ai.agents.base import BaseAgent
from co_ai.analysis.reflection_delta import compute_pipeline_delta
from co_ai.constants import GOAL
from co_ai.models.reflection_delta import ReflectionDeltaORM


class ReflectionDeltaAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        goal = self.memory.goals.get_or_create(context.get(GOAL))
        if not goal:
            self.logger.log("ReflectionDeltaSkipped", {"reason": "no goal in context"})
            return context
        runs = self.memory.pipeline_runs.get_by_goal_id(goal.id)

        if len(runs) < 2:
            self.logger.log("ReflectionDeltaSkipped", {
                "goal": goal,
                "reason": "only one or zero runs"
            })
            return context

        logged_deltas = 0
        for i, run_a in enumerate(runs):
            for run_b in runs[i+1:]:
                scores_a = self.memory.evaluations.get_by_run_id(run_a.run_id)
                scores_b = self.memory.evaluations.get_by_run_id(run_b.run_id)

                if not scores_a or not scores_b:
                    continue  # skip unscored runs

                delta = compute_pipeline_delta(run_a, run_b, scores_a, scores_b)

                self.memory.reflection_deltas.insert(ReflectionDeltaORM(**delta))
                self.logger.log("ReflectionDeltaLogged", {
                    "goal_id": goal.id,
                    "run_id_a": run_a.run_id,
                    "run_id_b": run_b.run_id,
                    "score_delta": delta.get("score_delta"),
                    "causal": delta.get("causal_improvement")
                })
                logged_deltas += 1

        context["reflection_deltas_logged"] = logged_deltas
        return context
