# co_ai/agents/ranking.py
import itertools
import random
import re

from co_ai.agents.base import BaseAgent


class RankingAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.elo_scores = {}
        self.preferences = cfg.get("preferences", ["novelty", "feasibility"])
        self.win_history = {}

    async def run(self, context: dict) -> dict:
        reviewed = context.get("reviewed", [])
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
            context["ranked"] = [(item["hypothesis"], 1000) for item in structured]
            self.log("RankSipped not enough ranked items to work with")
            return context

        self._initialize_scores(structured)
        self._rank_pairwise(structured)

        ranked = sorted(self.elo_scores.items(), key=lambda x: x[1], reverse=True)
        self.logger.log("TournamentCompleted", {
            "total_hypotheses": len(ranked),
            "win_loss_patterns": self._extract_win_loss_feedback()
        })

        context["ranked"] = ranked
        self.logger.log("RankedHypotheses", {
            "ranked_hypotheses": ranked,
            "preferences": self.preferences
        })
        return context

    def _extract_win_loss_feedback(self):
        """Return summary of which hypotheses won most often"""
        win_counts = {}

        for hyp, score in self.elo_scores.items():
            wins = sum(1 for a, b in self.win_history if a == hyp or b == hyp)
            win_counts[hyp] = wins

        return {
            "top_performers": [
                {"hypothesis": h, "wins": w}
                for h, w in sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
            ]
        }

    def _initialize_scores(self, reviewed):
        for item in reviewed:
            hyp = item["hypothesis"]
            self.elo_scores[hyp] = 1000

    def _rank_pairwise(self, reviewed):
        pairs = list(itertools.combinations(reviewed, 2))
        if not pairs:
            return

        # Limit number of comparisons per round
        comparisons = random.sample(pairs, k=min(self.cfg.get("max_comparisons", 6), len(pairs)))

        for item1, item2 in comparisons:
            hyp1 = item1["hypothesis"]
            hyp2 = item2["hypothesis"]

            prompt = self.prompt_template.format(
                goal=self.cfg.get("goal", "Unknown"),
                preferences=", ".join(self.preferences),
                hypothesis_a=hyp1,
                hypothesis_b=hyp2
            )

            self.log(f"Comparing:\n{hyp1[:60]}...\nvs\n{hyp2[:60]}...")

            try:
                response = self.call_llm(prompt).strip()
                match = re.search(r"better hypothesis:<([AB])>", response, re.IGNORECASE)

                if match:
                    winner = match.group(1).upper()
                    self._update_elo(hyp1, hyp2, winner)
                else:
                    self.logger.log("ComparisonParseFailed", {
                        "prompt_snippet": prompt[:200],
                        "response_snippet": response[:300]
                    })
            except Exception as e:
                self.logger.log("ComparisonError", {
                    "error": str(e),
                    "hypotheses": [hyp1[:100], hyp2[:100]]
                })

    def _update_elo(self, hyp1, hyp2, winner):
        K = 32
        R1 = 10 ** (self.elo_scores[hyp1] / 400)
        R2 = 10 ** (self.elo_scores[hyp2] / 400)
        E1 = R1 / (R1 + R2)
        E2 = R2 / (R1 + R2)

        S1 = 1 if winner == "A" else 0
        S2 = 1 - S1

        self.elo_scores[hyp1] = max(100, min(2800, self.elo_scores[hyp1] + K * (S1 - E1)))
        self.elo_scores[hyp2] = max(100, min(2800, self.elo_scores[hyp2] + K * (S2 - E2)))

        self.memory.store_ranking(hyp1, self.elo_scores[hyp1])
        self.memory.store_ranking(hyp2, self.elo_scores[hyp2])

        self.log(f"Elo updated: {hyp1[:50]}... → {self.elo_scores[hyp1]:.1f}")
        self.log(f"{hyp2[:50]}... → {self.elo_scores[hyp2]:.1f}")