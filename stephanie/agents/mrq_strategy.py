from omegaconf import OmegaConf

from stephanie.agents.base_agent import BaseAgent

DEFAULT_PIPELINES = [
    ["generation", "judge"],
    ["generation", "verifier", "judge"],
    ["generation", "reviewer", "judge"],
    ["cot_generator", "reviewer", "judge"],
    ["retriever", "generation", "judge"],
    ["retriever", "cot_generator", "judge"],
    ["retriever", "generation", "verifier", "judge"]
]

class MRQStrategyAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        # Load candidate strategies
        file_path = cfg.get("strategy_file")
        if file_path:
            strategy_cfg = OmegaConf.load(file_path)
            self.candidate_strategies = strategy_cfg.get("candidate_strategies", [])
        else:
            self.candidate_strategies = cfg.get("candidate_strategies", [])

        # Initialize model
        self.trained_ranker = None
        self.training_data = []

        # Attempt training (will be partial if data is incomplete)
        self.train_from_reflection_deltas()


    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {})
        goal_text = goal.get("goal_text", "")

        scored = []
        for pipeline in self.cfg.get("candidate_strategies", DEFAULT_PIPELINES):
            s = self.trained_ranker(pipeline)
            scored.append((pipeline, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]

        context["mrq_suggested_pipeline"] = best
        self.logger.log(
            "MRQPipelineSuggested",
            {"goal": goal_text, "suggested": best, "scored_candidates": scored},
        )

        return context

    def train_from_reflection_deltas(self):
        deltas = self.memory.reflection_deltas.get_all()
        examples = []

        for d in deltas:
            a = d.pipeline_a
            b = d.pipeline_b
            score_a = d.score_a
            score_b = d.score_b

            if not isinstance(a, list) or not isinstance(b, list):
                continue
            if score_a is None or score_b is None:
                continue
            if abs(score_a - score_b) < 0.05:
                continue

            label = "b" if score_b > score_a else "a"
            examples.append({
                "goal_text": d.get("goal_text"),
                "pipeline_a": a,
                "pipeline_b": b,
                "score_a": score_a,
                "score_b": score_b,
                "label": label
            })

        self.training_data = examples
        self.logger.log("MRQTrainingDataLoaded", {"count": len(examples)})

        # Train dummy ranker
        self.trained_ranker = self.symbolic_ranker()

    def symbolic_ranker(self):
        """
        Simple ranker that scores pipelines based on symbolic features.
        Prefers longer pipelines and known strong agents.
        """
        def score(pipeline):
            return (
                len(pipeline)
                + 1.5 * ("verifier" in pipeline)
                + 1.2 * ("reviewer" in pipeline)
                + 1.0 * ("retriever" in pipeline)
                + 0.8 * ("cot_generator" in pipeline)
            )
        return score