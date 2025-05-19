from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL
import re

class JudgeAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.max_iterations = cfg.get("max_turns", 3)  # Debate depth
        self.strategy = cfg.get("strategy", "scientific_debate")

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        hypotheses = context.get(self.input_key, [])

        if len(hypotheses) < 2:
            self.logger.log("NotEnoughHypotheses", {"count": len(hypotheses)})
            return context

        # Run pairwise comparisons
        rankings = []
        for i in range(0, len(hypotheses), 2):
            try:
                result = self._compare_pair(
                    context, hypotheses[i], hypotheses[i + 1]
                )
                rankings.append(result)
            except IndexError:
                # Odd number of hypotheses — last one gets free pass
                rankings.append({"winner": hypotheses[-1], "reason": "odd_count"})
                break

        context[self.output_key] = rankings
        self.logger.log(
            "GeneratedRankings",
            {"goal_snippet": goal[:60], "ranking_count": len(rankings)},
        )

        return context

    def _compare_pair(self, context, hypothesis_a, hypothesis_b):
        """Compare two hypotheses and return which is better"""
        to_merge = {
            "hypothesis_a": hypothesis_a,
            "hypothesis_b": hypothesis_b,
            "reflection_a": context.get("reflection_a", ""),
            "reflection_b": context.get("reflection_b", ""),
            "notes": context.get("comparison_notes", ""),
        }

        prompt = self.prompt_loader.load_prompt(self.cfg, {
                **context,
                **to_merge
            })
        response = self.call_llm(prompt, context)

        winner_match = re.search(r"better hypothesis:<([AB])>", response, re.IGNORECASE)
        reason_match = re.search(r"reason:<(.+)>", response, re.DOTALL)

        winner = winner_match.group(1).upper() if winner_match else "A"
        reason = reason_match.group(1).strip() if reason_match else "No clear winner"

        return {
            "hypothesis_a": hypothesis_a[:100] + "...",
            "hypothesis_b": hypothesis_b[:100] + "...",
            "winner": winner,
            "reason": reason,
            "prompt_used": prompt[:500] + "...",
        }

    def _compare_all_pairs(self, context, hypotheses):
        """Compare all possible pairs among N hypotheses"""
        scores = [0] * len(hypotheses)

        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                result = self._compare_pair(context, hypotheses[i], hypotheses[j])
                if result["winner"] == "A":
                    scores[i] += 1
                elif result["winner"] == "B":
                    scores[j] += 1

        # Return ranked indices
        return sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)