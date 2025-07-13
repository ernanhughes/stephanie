# stephanie/agents/symbolic_optimizer.py
from collections import defaultdict

from stephanie.agents import BaseAgent
from stephanie.constants import GOAL, PIPELINE
from stephanie.memory.symbolic_rule_store import SymbolicRuleORM


class SymbolicOptimizerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.score_target = cfg.get("score_target", "correctness")
        self.min_scores = cfg.get("min_scores", 2)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, {})
        goal_type = goal.get("goal_type", "unknown")

        # Step 1: Retrieve score history for this goal type
        score_history = self.memory.evaluations.get_by_goal_type(goal_type)

        # Step 2: Analyze pipelines
        best_pipeline = self.find_best_pipeline(score_history)

        if best_pipeline:
            rule_dict = {
                "target": "pipeline",
                "filter": {"goal_type": goal_type},
                "attributes": {"pipeline": best_pipeline},
                "source": "symbolic_optimizer",
            }

            context["symbolic_suggestion"] = rule_dict

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

            self.logger.log(
                "SymbolicPipelineSuggestion",
                {
                    "goal_type": goal_type,
                    "suggested_pipeline": best_pipeline,
                    "score_type": self.score_target,
                },
            )

        return context

    def find_best_pipeline(self, score_history):
        scores_by_pipeline = defaultdict(list)

        for score in score_history:
            run_id = score.get("run_id")
            if not run_id:
                continue
            pipeline_run = self.memory.pipeline_runs.get_by_run_id(run_id)
            if not pipeline_run or not pipeline_run.pipeline:
                continue

            str_pipeline = str(pipeline_run.pipeline)
            score_val = score.get("score")
            if score_val is not None:
                scores_by_pipeline[str_pipeline].append(score_val)

        # Only keep pipelines with enough data
        pipeline_scores = {
            pipe: sum(vals) / len(vals)
            for pipe, vals in scores_by_pipeline.items()
            if len(vals) >= self.min_scores
        }

        self.logger.log(
            "PipelineScoreSummary",
            {
                "score_type": self.score_target,
                "pipeline_scores": {
                    pipe: round(avg, 4) for pipe, avg in pipeline_scores.items()
                },
            },
        )

        if not pipeline_scores:
            return None

        best = max(pipeline_scores.items(), key=lambda x: x[1])
        return list(best[0])  # convert stringified list back to list
