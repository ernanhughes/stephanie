# co_ai/agents/ranking.py
import itertools
import random
import re
from typing import Optional

from co_ai.agents.base import BaseAgent

class RankingAgent(BaseAgent):
    """
    The Ranking agent simulates scientific debate between hypotheses using a tournament-style approach.
    
    From the paper:
    > 'The Ranking agent employs an Elo-based tournament to assess and prioritize generated hypotheses'
    """
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.elo_scores = {}
        self.strategy = cfg.get("strategy", "debate")
        self.max_comparisons = cfg.get("max_comparisons", 6)
        self.win_history = []
        self.preferences = cfg.get("preferences", ["novelty", "feasibility"])


    async def run(self, context: dict) -> dict:
        """
        Rank hypotheses using pairwise comparisons and Elo updates.
        
        Args:
            context: Dictionary with keys:
                - hypotheses: list of hypothesis strings
                - goal: research objective
                - preferences: override criteria
        """
        hypotheses = context.get("hypotheses", [])

        if len(hypotheses) < 2:
            self.logger.log("NotEnoughHypothesesForRanking", {
                "count": len(hypotheses),
                "reason": "less than 2 hypotheses"
            })
            context["ranked"] = [(h, 1000) for h in hypotheses]
            return context

        self._initialize_elo(hypotheses)

        pairs = list(itertools.combinations(hypotheses, 2))
        comparisons = random.sample(pairs, k=min(self.max_comparisons, len(pairs)))

        for hyp1, hyp2 in comparisons:
            prompt = self._build_ranking_prompt(hyp1, hyp2, context)
            response = self.call_llm(prompt).strip()
            winner = self._parse_response(response)

            if winner:
                self._update_elo(hyp1, hyp2, winner)
            else:
                self.logger.log("ComparisonParseFailed", {
                    "prompt_snippet": prompt[:200],
                    "response_snippet": response[:300],
                    "agent": self.__class__.__name__
                })

        ranked = sorted(self.elo_scores.items(), key=lambda x: x[1], reverse=True)
        context["ranked"] = ranked
        context["preferences_used"] = self.preferences
        self.logger.log("TournamentCompleted", {
            "total_hypotheses": len(ranked),
            "win_loss_patterns": self._extract_win_loss_feedback(),
            "preferences": self.preferences
        })
        return context

    def _initialize_elo(self, hypotheses):
        for h in hypotheses:
            if h not in self.elo_scores:
                self.elo_scores[h] = 1000

    def _build_ranking_prompt(self, hyp1, hyp2, context):
        """Build prompt dynamically with or without reviews."""
        return self.prompt_loader.load_prompt(self.cfg, {**context,  **{"hypothesis_a":hyp1, "hypothesis_b":hyp2}})

    def _conduct_multi_turn_debate(self, goal, hyp1, hyp2, turns=3):
        """Simulate multi-turn scientific debate between hypotheses"""
        for i in range(turns):
            prompt = self._build_ranking_prompt(goal, hyp1, hyp2)
            response = self.call_llm(prompt).strip()
            winner = self._parse_response(response)
            if winner:
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
                {"hypotheses": h, "wins": w}
                for h, w in sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "total_matches": len(self.win_history),
            "preferences_used": self.preferences
        }

    def _rank_pairwise(self, reviewed, context):
        pairs = list(itertools.combinations(reviewed, 2))
        if not pairs:
            return

        # Limit number of comparisons per round
        comparisons = random.sample(pairs, k=min(self.cfg.get("max_comparisons", 6), len(pairs)))

        for item1, item2 in comparisons:
            hyp1 = item1["hypotheses"]
            hyp2 = item2["hypotheses"]

            merged = {**self.cfg, **{"hypothesis_a": hyp1, "hypothesis_b": hyp2}}


            prompt = self.prompt_loader.load_prompt(merged, context=context)

            self.log(f"Comparing:\n{hyp1[:60]}...\nvs\n{hyp2[:60]}...")

            try:
                response = self.call_llm(prompt).strip()
                winner = self._parse_response(response)

                if winner:
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
        K = self.cfg.get("elo_k", 32)
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

    def _parse_response(self, response: str) -> Optional[str]:
        """
        Try multiple methods to extract winner from LLM output
        
        Returns:
            'A' or 'B' based on comparison
        """
        # Try matching structured formats first
        structured_match = re.search(r"better[\s_]?hypothesis[^\w]*([AB12])", response, re.IGNORECASE)
        if structured_match:
            winner_key = structured_match.group(1).upper()
            return "A" if winner_key in ("A", "1") else "B"

        # Try matching natural language statements
        lang_match = re.search(r"(?:prefer|choose|recommend|select)(\s+idea|\s+hypothesis)?[:\s]+([AB12])", response, re.IGNORECASE)
        if lang_match:
            winner_key = lang_match.group(2).upper()
            return "A" if winner_key in ("A", "1") else "B"

        # Try matching conclusion phrases
        conclusion_match = re.search(r"conclude[d]?\s+with\s+better[\s_]idea:\s*(\d)", response, re.IGNORECASE)
        if conclusion_match:
            winner_key = conclusion_match.group(1)
            return "A" if winner_key == "1" else "B"

        # Default fallback logic
        print("[⚠️] Could not extract winner from response.")
        self.logger.log("ParseError", {
                    "error": "Could not extract winner from response",
                    "response": response
                })
        return None