from co_ai.agents import BaseAgent
from co_ai.models import HypothesisORM
from co_ai.scoring.proximity import ProximityScore


class ScorerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.weight_proximity = cfg.get("weight_proximity", 0.4)
        self.weight_review = cfg.get("weight_review", 0.3)
        self.weight_llm_judge = cfg.get("weight_llm_judge", 0.2)
        self.weight_elo = cfg.get("weight_elo", 0.1)
        print(f"ScorerAgent initialized with weights: {self.weight_proximity}, {self.weight_review}, {self.weight_llm_judge}, {self.weight_elo}")

    def score(self, hypothesis: HypothesisORM, context: dict) -> float:
        # Extract features (assumes values already normalized 0–1)
        s = ProximityScore(self.cfg, self.memory, self.logger)
        proximity = s.get_score(hypothesis, context)
        review = hypothesis.review or 0.0
        llm_judge = hypothesis.reflection or 0.0
        elo = hypothesis.elo_rating or 0.0

        # Weighted score aggregation
        score = (
            self.weight_proximity * proximity +
            self.weight_review * review +
            self.weight_llm_judge * llm_judge +
            self.weight_elo * elo
        )

        return round(score, 4)

    async def run(self, context: dict):
        hypotheses = self.get_hypotheses(context)
        scored = []
        scores = []
        for hypo in hypotheses:
            h = self.memory.hypotheses.get_from_text(hypo)
            score = self.score(h, context)
            scores.append(score)

            self.logger.log("HypothesisScored", {
                "prompt": h.prompt_id,
                "hypothesis": h.text,
                "score": score
            })

            scored.append(hypo)

        context["scored_hypotheses"] = scored
        return context
