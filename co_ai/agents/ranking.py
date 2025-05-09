# co_ai/agents/ranking.py
from co_ai.agents.base import BaseAgent
import math
from collections import defaultdict

class RankingAgent(BaseAgent):
    def __init__(self, memory=None, logger=None):
        super().__init__(memory=memory, logger=logger)
        self.default_score = 1000

    async def run(self, input_data: dict) -> dict:
        reviews = input_data.get("reviews", [])
        self.log(f"Ranking {len(reviews)} structured reviews...")

        grouped = defaultdict(list)
        for item in reviews:
            grouped[item["hypothesis"]].append(item["review"])

        elo_scores = {hyp: self.default_score for hyp in grouped}

        # Pairwise comparisons
        hypotheses = list(grouped.keys())
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                a, b = hypotheses[i], hypotheses[j]
                a_score = self.heuristic_score(grouped[a])
                b_score = self.heuristic_score(grouped[b])

                if a_score > b_score:
                    self._update_scores(elo_scores, a, b)
                else:
                    self._update_scores(elo_scores, b, a)

        ranked = sorted(elo_scores.items(), key=lambda x: x[1], reverse=True)
        self.log("Ranking complete.")
        return {
            "ranked": [{"hypothesis": hyp, "elo": elo_scores[hyp]} for hyp, _ in ranked]
        }

    def heuristic_score(self, reviews):
        return sum(len(r) for r in reviews)

    def _update_scores(self, scores, winner, loser, k=32):
        rating_winner = scores[winner]
        rating_loser = scores[loser]
        expected_win = 1 / (1 + 10 ** ((rating_loser - rating_winner) / 400))
        scores[winner] += int(k * (1 - expected_win))
        scores[loser] -= int(k * expected_win)
