from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, REFLECTION
import re


class JudgeAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.strategy = cfg.get("strategy", "scientific_debate")
        self.max_hypotheses = cfg.get("max_compare", 3)  # Max hypotheses per round
        self.preference_weights = {
            "goal_consistency": 1.2,
            "novelty": 1.5,
            "feasibility": 1.0,
            "biological_plausibility": 1.7,
            "simplicity": 1.0,
        }

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        hypotheses = context.get(self.input_key, [])

        if len(hypotheses) < 2:
            self.logger.log("NotEnoughHypotheses", {"count": len(hypotheses)})
            return context

        # Get reflections for all hypotheses
        reflections = []
        for h in hypotheses:
            reflection = context.get(REFLECTION, {})
            reflections.append(reflection.get(h, {}))

        # Run comparisons based on size of hypotheses
        length = len(hypotheses)
        if length < 7:
            rankings = await self._run_all_pair(context, hypotheses, reflections)
        elif length < 11:
            rankings = await self._run_tournament(context, hypotheses, reflections)
        elif length < 16:
            rankings = await self._run_top_k(context, hypotheses, reflections)
        else:
            rankings = await self._run_default(context, hypotheses, reflections)

        # Store results
        self._log_rankings(goal, rankings)

        # Update context with winner
        best_idx = rankings.index(max(rankings, key=lambda x: x["score"]))
        context["best_hypothesis"] = hypotheses[best_idx]
        context[self.output_key] = rankings

        return context

    async def _run_all_pair(self, context, hypotheses, reflections):
        """Compare every hypothesis against every other (for small batches)"""
        scores = [0] * len(hypotheses)

        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                result = await self._compare_pair(
                    context,
                    hypotheses[i],
                    hypotheses[j],
                    reflections[i],
                    reflections[j],
                )
                if result["winner"] == "A":
                    scores[i] += 1
                elif result["winner"] == "B":
                    scores[j] += 1

        return [
            {"index": idx, "score": score, "text": hypotheses[idx]}
            for idx, score in enumerate(scores)
        ]

    async def _run_tournament(self, context, hypotheses, reflections):
        """Run multi-round elimination-style judging"""
        current_round = list(range(len(hypotheses)))
        scores = [0] * len(hypotheses)

        while len(current_round) > 1:
            next_round = []

            for i in range(0, len(current_round) - 1, 2):
                a = current_round[i]
                b = current_round[i + 1]

                result = await self._compare_pair(
                    context,
                    hypotheses[a],
                    hypotheses[b],
                    reflections[a],
                    reflections[b],
                )

                winner = a if result["winner"] == "A" else b
                loser = b if result["winner"] == "A" else a

                scores[winner] += 1
                next_round.append(winner)

            if len(current_round) % 2 == 1:
                next_round.append(current_round[-1])

            current_round = next_round

        return [
            {"index": idx, "score": scores[idx], "text": hypotheses[idx]}
            for idx in current_round
        ]

    async def _run_top_k(self, context, hypotheses, reflections, k=3):
        """Use composite scoring to select top k before judging"""
        scored = []
        for i, h in enumerate(hypotheses):
            score = self._compute_composite_score(reflections[i])
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_indices = [x[0] for x in scored[:k]]
        top_hypotheses = [hypotheses[i] for i in top_indices]
        top_reflections = [reflections[i] for i in top_indices]

        # Compare top candidates
        rankings = await self._run_all_pair(context, top_hypotheses, top_reflections)

        return rankings

    async def _run_default(self, context, hypotheses, reflections):
        """Default behavior: compare in pairs until one remains"""
        scores = [0] * len(hypotheses)
        for i in range(0, len(hypotheses), 2):
            try:
                result = await self._compare_pair(
                    context,
                    hypotheses[i],
                    hypotheses[i + 1],
                    reflections[i],
                    reflections[i + 1],
                )
                winner = i if result["winner"] == "A" else i + 1
                scores[winner] += 1
            except IndexError:
                # Odd number of hypotheses
                scores[-1] += 1
                break

        return [
            {"index": idx, "score": scores[idx], "text": hypotheses[idx]}
            for idx in range(len(hypotheses))
        ]

    async def _compare_pair(
        self, context, hypothesis_a, hypothesis_b, reflection_a, reflection_b
    ):
        """Ask LLM to compare two hypotheses using structured review"""
        to_merge = {
            "hypothesis_a": hypothesis_a,
            "hypothesis_b": hypothesis_b,
            "reflection_a": reflection_a.get("full_review", ""),
            "reflection_b": reflection_b.get("full_review", ""),
            "notes": context.get("comparison_notes", ""),
        }
        merged = {**context, **to_merge}

        prompt = self.prompt_loader.load_prompt(
            self.cfg, merged
        )
        response = self.call_llm(prompt, context)

        # Extract winner
        winner_match = re.search(r"better hypothesis:<([AB])>", response, re.IGNORECASE)
        reason_match = re.search(r"reason:<(.+)>", response, re.DOTALL)

        winner = winner_match.group(1).upper() if winner_match else "A"
        reason = reason_match.group(1).strip() if reason_match else "No clear winner"

        return {
            "winner": winner,
            "reason": reason,
            "prompt_used": prompt[:500] + "...",
            "hypothesis_a_snippet": hypothesis_a[:200],
            "hypothesis_b_snippet": hypothesis_b[:200],
        }

    def _compute_composite_score(self, reflection):
        """Compute weighted score based on reflection and preference alignment"""
        base_score = reflection.get("elo_rating", 1000) / 10

        correctness = reflection.get("correctness_score", 3) * 10
        novelty = reflection.get("novelty_score", 3) * 10
        feasibility = reflection.get("feasibility_score", 3) * 10

        total = base_score + correctness + novelty + feasibility

        return total

    def _log_rankings(self, goal: str, rankings: list):
        """Log each hypothesis's score and reasoning"""
        for item in rankings:
            self.logger.log(
                "HypothesisRanked",
                {
                    "goal_snippet": goal[:60],
                    "hypothesis_snippet": item["text"][:100],
                    "score": item["score"],
                },
            )