
from collections import defaultdict

from co_ai.agents import BaseAgent
from co_ai.constants import GOAL, PIPELINE
from co_ai.memory.symbolic_rule_store import SymbolicRuleORM


class SymbolicOptimizerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, {})
        goal_type = goal.get("goal_type", "unknown")

        # Retrieve score history for this goal type
        score_history = self.memory.scores.get_by_goal_type(goal_type)

        # Analyze past scores to determine best-performing pipeline
        best_pipeline = self.find_best_pipeline(score_history)

        if best_pipeline:
            rule_dict = {
                "target": "pipeline",
                "filter": {"goal_type": goal_type},
                "attributes": {"pipeline": best_pipeline},
                "source": "optimizer",
            }

            context["symbolic_suggestion"] = rule_dict

            # Optional: persist it
            if self.cfg.get("auto_write_rules", False):
                existing = self.memory.symbolic_rules.find_matching_rule(
                    target="pipeline",
                    filter={"goal_type": goal_type},
                    attributes={"pipeline": best_pipeline},
                )
                if not existing:
                    new_rule = SymbolicRuleORM.from_dict(rule_dict)
                    self.memory.symbolic_rules.insert(new_rule)
                    self.logger.log("SymbolicRuleAutoCreated", rule_dict)

            self.logger.log("SymbolicPipelineSuggestion", {
                "goal_type": goal_type,
                "suggested_pipeline": best_pipeline
            })

        return context

    def find_best_pipeline(self, score_history):
        scores_by_pipeline = defaultdict(list)

        for score in score_history:
            run_id = score.get("run_id")
            if run_id:
                pipeline_run = self.memory.pipeline_runs.get_by_run_id(run_id)
                pipeline = pipeline_run.pipeline
                str_pipeline = str(pipeline)
                score_val = score.get("score")
                if str_pipeline and score_val is not None:
                    scores_by_pipeline[str_pipeline].append(score_val)

        pipeline_scores = {
            pipe: sum(vals) / len(vals)
            for pipe, vals in scores_by_pipeline.items()
            if len(vals) >= 2
        }

        self.logger.log(
            "PipelineScoreSummary",
            {
                "pipeline_scores": {
                    pipe: round(avg, 4) for pipe, avg in pipeline_scores.items()
                }
            },
        )

        if not pipeline_scores:
            return None

        best = max(pipeline_scores.items(), key=lambda x: x[1])
        return list(best[0])
