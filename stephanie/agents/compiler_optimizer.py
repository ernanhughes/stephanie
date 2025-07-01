import statistics
from collections import defaultdict

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, PIPELINE_RUN_ID


class CompilerOptimizerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.score_threshold = cfg.get("score_threshold", 5.0)  # optional tuning param

    async def run(self, context: dict) -> dict:
        pipeline_run_id = context.get(PIPELINE_RUN_ID)
        goal = context.get(GOAL)

        # Step 1: Fetch all hypotheses and their scores for this run
        hypotheses = self.memory.hypotheses.get_by_pipeline_run(pipeline_run_id)
        all_scores = self.memory.evaluations.get_by_pipeline_run(pipeline_run_id)

        # Group scores by prompt_id
        prompt_scores = defaultdict(list)
        for score in all_scores:
            if score.prompt_id:
                prompt_scores[score.prompt_id].append(score)

        summary = []

        # Step 2: Analyze performance per prompt
        for prompt_id, scores in prompt_scores.items():
            prompt = self.memory.prompt.get(prompt_id)
            raw_scores = [s.score for s in scores if s.score is not None]

            if not raw_scores:
                continue

            avg = statistics.mean(raw_scores)
            std = statistics.stdev(raw_scores) if len(raw_scores) > 1 else 0
            count = len(raw_scores)
            high_score_rate = sum(s >= self.score_threshold for s in raw_scores) / count

            summary.append({
                "prompt_id": prompt_id,
                "prompt_text": prompt.prompt_text[:100] if prompt else "<unknown>",
                "avg_score": avg,
                "std_dev": std,
                "count": count,
                "high_score_rate": round(high_score_rate * 100, 2)
            })

        # Step 3: Log or save insights
        top_prompts = sorted(summary, key=lambda x: x["avg_score"], reverse=True)[:5]
        self.logger.log("CompilerOptimizerSummary", {
            "goal": goal.get("goal_text"),
            "top_prompts": top_prompts,
            "pipeline_run_id": pipeline_run_id
        })

        print("\n=== Top Performing Compiled Prompts ===")
        for i, p in enumerate(top_prompts):
            print(f"[{i+1}] Avg Score: {p['avg_score']:.2f} | Used {p['count']}x | Prompt: {p['prompt_text']}")

        # Step 4 (future): Update strategy weights, rules, DSPy prior preferences

        context["compiler_optimization_summary"] = summary
        return context
