from co_ai.agents import BaseAgent
from co_ai.logs.icons_enum import get_event_icon

class HypothesisScorerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.weight_proximity = cfg.get("weight_proximity", 0.4)
        self.weight_review = cfg.get("weight_review", 0.3)
        self.weight_llm_judge = cfg.get("weight_llm_judge", 0.2)
        self.weight_elo = cfg.get("weight_elo", 0.1)

    def score(self, hypothesis: dict, context: dict) -> float:
        # Extract features (assumes values already normalized 0–1)
        proximity = hypothesis.get("proximity_score", 0.0)
        review = hypothesis.get("review_score", 0.0)
        llm_judge = hypothesis.get("llm_judge_score", 0.0)
        elo = hypothesis.get("elo_rating", 0.0) / 1000.0  # normalize

        # Weighted score aggregation
        score = (
            self.weight_proximity * proximity +
            self.weight_review * review +
            self.weight_llm_judge * llm_judge +
            self.weight_elo * elo
        )

        return round(score, 4)

    async def run(self, context: dict):
        hypotheses = context.get("hypotheses", [])
        scored = []

        for hypo in hypotheses:
            score = self.score(hypo, context)
            hypo["composite_score"] = score

            self.logger.log("HypothesisScored", {
                "prompt": hypo.get("prompt_text"),
                "hypothesis": hypo.get("text"),
                "score": score
            })

            scored.append(hypo)

        context["scored_hypotheses"] = scored
        return context
