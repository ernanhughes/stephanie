# co_ai/agents/ranking.py
import itertools
import random
import re

from co_ai.agents.base import BaseAgent
from co_ai.utils.prompt_loader import load_prompt_from_file

class RankingAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.elo_scores = {}
        self.win_history = {}
        self.preferences = cfg.get("preferences", ["novelty", "feasibility"])


    async def run(self, context: dict) -> dict:
        hypotheses = context.get("hypotheses", [])
        reflections = context.get("reflections", [])

        if len(hypotheses) < 2:
            self.logger.log("NotEnoughHypothesesForRanking", {
                "count": len(hypotheses),
                "reason": "fewer than 2 hypotheses"
            })
            context["ranked"] = [(h, 1000) for h in hypotheses]
            return context

        self._initialize_scores(hypotheses)

        # Build review map if available
        review_map = {r["hypothesis"]: r["review"] for r in reflections if "hypothesis" in r and "review" in r}

        for item1, item2 in self._generate_pairwise_comparisons(hypotheses):
            hyp1 = item1["hypothesis"] if isinstance(item1, dict) else item1
            hyp2 = item2["hypothesis"] if isinstance(item2, dict) else item2

            prompt = self._build_ranking_prompt(context["goal"], hyp1, hyp2, review_map.get(hyp1), review_map.get(hyp2))
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

        ranked = sorted(self.elo_scores.items(), key=lambda x: x[1], reverse=True)
        context["ranked"] = ranked
        self.logger.log("TournamentCompleted", {
            "total_hypotheses": len(ranked),
            "win_loss_patterns": self._extract_win_loss_feedback(),
            "preferences": self.preferences
        })
        return context

    def _build_ranking_prompt(self, goal, hyp1, hyp2, review1=None, review2=None):
        """Build prompt dynamically with or without reviews."""
        return self.prompt_template.format(
            goal=goal,
            preferences=", ".join(self.preferences),
            hypothesis_a=hyp1,
            hypothesis_b=hyp2,
            review_a=review1 or "No prior review",
            review_b=review2 or "No prior review"
        )

    def _conduct_multi_turn_debate(self, goal, hyp1, hyp2, turns=3):
        """Simulate multi-turn scientific debate between hypotheses"""
        for i in range(turns):
            prompt = self._build_ranking_prompt(goal, hyp1, hyp2)
            response = self.call_llm(prompt).strip()
            match = re.search(r"better hypothesis:<([AB])>", response, re.IGNORECASE)
            if match:
                winner = match.group(1).upper()
                self._update_elo(hyp1, hyp2, winner)
            else:
                break


    def _generate_pairwise_comparisons(self, hypotheses):
        """Generate combinations of hypothesis pairs for ranking"""
        return itertools.combinations(hypotheses, 2)

    def _generate_proximity_based_pairs(self, hypotheses):
        """Prioritize comparisons between similar hypotheses"""
        similarities = [
            (h1, h2, self._compute_similarity(h1, h2)) 
            for h1, h2 in itertools.combinations(hypotheses, 2)
        ]
        return sorted(similarities, key=lambda x: x[2], reverse=True)

    def _extract_win_loss_feedback(self):
        """Return summary of which hypotheses won most often"""
        win_counts = {}

        for hyp1, hyp2, winner in self.win_history:
            winner_hypothesis = hyp1 if winner == "A" else hyp2
            win_counts[winner_hypothesis] = win_counts.get(winner_hypothesis, 0) + 1

        return {
            "top_performers": [
                {"hypothesis": h, "wins": w}
                for h, w in sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "total_matches": len(self.win_history),
            "preferences_used": self.preferences
        }

    def _initialize_scores(self, hypotheses):
        for h in hypotheses:
            if h not in self.elo_scores:
                self.elo_scores[h] = 1000
                
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
        K = self.cfg.agent.get("elo_k", 32)
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

        self.win_history.append((hyp1, hyp2, winner))
        self.logger.log("RankingUpdated", {
            "hypothesis_a": hyp1,
            "hypothesis_b": hyp2,
            "winner": winner,
            "elo_a": self.elo_scores[hyp1],
            "elo_b": self.elo_scores[hyp2]
        })  
