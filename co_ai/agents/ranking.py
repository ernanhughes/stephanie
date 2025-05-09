# co_ai/agents/ranking.py
import itertools
import random

from co_ai.agents.base import BaseAgent


class RankingAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.elo_scores = {}

    async def run(self, input_data: dict) -> dict:
        reviewed = input_data.get("reviewed", [])
        structured = [
            {
                "hypothesis": item["hypothesis"],
                "review": item["review"],
                "persona": item.get("persona", "Neutral")
            }
            for item in reviewed
        ]

        if len(structured) < 2:
            # Not enough for ELO, return baseline
            return {"ranked": [(item["hypothesis"], 1000) for item in structured]}

        self._initialize_scores(structured)
        self._rank_pairwise(structured)

        ranked = sorted(self.elo_scores.items(), key=lambda x: x[1], reverse=True)
        return {"ranked": ranked}

    def _initialize_scores(self, reviewed):
        for item in reviewed:
            hyp = item["hypothesis"]
            self.elo_scores[hyp] = 1000

    def _rank_pairwise(self, reviewed):
        pairs = list(itertools.combinations(reviewed, 2))
        if not pairs:
            return

        comparisons = random.sample(pairs, k=min(6, len(pairs)))

        for item1, item2 in comparisons:
            prompt = (
                f"You are a critical analyst. Two hypotheses were reviewed as follows:\n\n"
                f"Hypothesis A:\n{item1['hypothesis']}\nReview: {item1['review']}\n\n"
                f"Hypothesis B:\n{item2['hypothesis']}\nReview: {item2['review']}\n\n"
                f"Which hypothesis is stronger overall and why? Just reply with 'A' or 'B'."
            )
            winner = self.call_llm(prompt).strip().upper()

            if winner not in {"A", "B"}:
                continue

            self._update_elo(item1["hypothesis"], item2["hypothesis"], winner)

    def _update_elo(self, hyp1, hyp2, winner):
        K = 32
        R1 = 10 ** (self.elo_scores[hyp1] / 400)
        R2 = 10 ** (self.elo_scores[hyp2] / 400)
        E1 = R1 / (R1 + R2)
        E2 = R2 / (R1 + R2)

        S1 = 1 if winner == "A" else 0
        S2 = 1 - S1

        self.elo_scores[hyp1] += K * (S1 - E1)
        self.elo_scores[hyp2] += K * (S2 - E2)
        self.memory.store_ranking(hyp1, self.elo_scores[hyp1])
        self.memory.store_ranking(hyp2, self.elo_scores[hyp2])  
        self.log(f"Updated ELO scores: {self.elo_scores}", structured=False)
        self.log(f"Ranked hypotheses: {self.elo_scores}", structured=False) 
