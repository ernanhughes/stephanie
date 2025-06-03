import re

from co_ai.constants import REFLECTION
from co_ai.models import ScoreORM
from co_ai.scoring.base_score import BaseScore


class ReflectionScore(BaseScore):
    name = REFLECTION
    default_value = 0.0

    def __init__(self, cfg, memory, logger=None):
        super().__init__(cfg, memory, logger)
        self.scores = {}

    def compute(self, hypothesis: dict[str, any], context: dict[str, any]) -> float:
        """
        Compute a composite score from multiple reflection dimensions.
        """
        reflection_text = hypothesis.get(REFLECTION)
        if not reflection_text:
            return self.default_value

        try:
            self.scores = self._extract_scores(reflection_text)
            if not self.scores:
                return self.default_value

            # Cache full dict for later analysis
            hypothesis["reflection_scores"] = self.scores

            # Save each dimension to score store
            for dimension, data in self.scores.items():
                self._store_score(hypothesis, context, dimension, data)

            # Return correctness if available, otherwise average
            return self.scores.get("correctness", {}).get("score") or self._composite_score(self.scores)

        except Exception as e:
            self.logger.log("ReflectionScoreParseFailed", {
                "hypothesis_id": hypothesis.get("id"),
                "error": str(e),
                "reflection_snippet": reflection_text[:300],
            })
            return self.default_value

    def _extract_scores(self, text: str) -> dict:
        """
        Extracts structured scores and rationales from reflection text.
        Returns a dictionary of dimension → {score, rationale}
        """
        pattern = r"#\s*(?P<dimension>\w+)\s*Assessment\s*Score:\s*(?P<score>\d+\.?\d*)\s*Reasoning:\s*(?P<rationale>.*?)(?=\n#|\Z)"
        matches = re.finditer(pattern, text, re.DOTALL)

        results = {}
        for m in matches:
            dimension = m.group("dimension").lower()
            try:
                score = float(m.group("score"))
            except ValueError:
                continue
            rationale = m.group("rationale").strip()
            results[dimension] = {"score": score, "rationale": rationale}

        # Include self-reward if found
        reward_match = re.search(r"# Self-Reward Score\s*Score\s*\[?(\d{1,3})\]?", text)
        if reward_match:
            results["reward"] = {"score": float(reward_match.group(1)), "rationale": "Self-reward score"}

        return results

    def _composite_score(self, scores: dict) -> float:
        if not scores:
            return self.default_value
        return sum(item["score"] for item in scores.values()) / len(scores)

    def _store_score(self, hypothesis: dict, context: dict, dimension: str, data: dict):
        score_obj = ScoreORM(
            goal_id=hypothesis.get("goal_id"),
            hypothesis_id=hypothesis.get("id"),
            agent_name=self.agent_name,
            model_name=self.model_name,
            evaluator_name="ReflectionScore",
            score_type=dimension,
            score=data["score"],
            rationale=data.get("rationale", ""),
            pipeline_run_id=context.get("pipeline_run_id"),
            metadata={"source": "structured_reflection"},
        )
        self.memory.scores.insert(score_obj)

    def get_score(self, hypothesis: dict, context: dict) -> float:
        """
        Alias for compute(...) — used for consistency with other score classes.
        """
        return self.compute(hypothesis, context)
