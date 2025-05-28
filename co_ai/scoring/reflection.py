from co_ai.scoring.base_score import BaseScore
from co_ai.constants import REFLECTION
from co_ai.models import ScoreORM
import re


class ReflectionScore(BaseScore):
    name = REFLECTION
    default_value = 0.0

    def __init__(self, cfg, memory, logger=None):
        super().__init__(cfg, memory, logger)

    def compute(self, hypothesis: dict[str, any], context: dict[str, any]) -> float:
        """
        Extracts a numerical score from a structured reflection summary.
        Returns a single score or composite if multiple are present.
        """
        reflection_text = hypothesis.get(REFLECTION)
        if not reflection_text:
            return self.default_value

        try:
            # Try to extract structured sections
            scores = self._extract_scores(reflection_text)

            if not scores:
                return self.default_value

            # Optionally combine all into a composite score
            composite = sum(s["score"] for s in scores.values()) / len(scores)
            return composite

        except Exception as e:
            self.logger.log("ReflectionScoreParseFailed", {
                "hypothesis_id": hypothesis.get("id"),
                "error": str(e),
                "reflection_snippet": reflection_text[:200]
            })
            return self.default_value

    def _extract_scores(self, reflection_text: str) -> dict:
        """Extracts multiple score blocks from reflection summary"""
        pattern = r"#\s*(?P<dimension>\w+)\s*Assessment\s*Score:\s*(?P<score>\d+\.?\d*)\s*Reasoning:\s*(?P<rationale>.*?)(?=\n#|\Z)"
        matches = re.finditer(pattern, reflection_text, re.DOTALL)

        results = {}
        for m in matches:
            dimension = m.group("dimension").lower()
            score = float(m.group("score"))
            rationale = m.group("rationale").strip()

            results[dimension] = {"score": score, "rationale": rationale}

        return results

    def get_score(self, hypothesis: dict, context: dict) -> float:
        reflection_text = hypothesis.get(REFLECTION)
        if not reflection_text:
            return self.default_value

        scores_dict = self._extract_scores(reflection_text)

        if not scores_dict:
            return self.default_value

        # Cache full dict for later analysis
        hypothesis["reflection_scores"] = scores_dict

        # Insert each score separately
        for dimension, data in scores_dict.items():
            score_obj = ScoreORM(
                goal_id=hypothesis.get("goal_id"),
                hypothesis_id=hypothesis.get("id"),
                agent_name=self.agent_name,
                model_name=self.model_name,
                evaluator_name="ReflectionDeltaAgent",
                score_type=dimension,
                score=data["score"],
                rationale=data["rationale"],
                run_id=context.get("run_id"),
                metadata={"source": "structured_reflection"},
            )
            self.memory.scores.insert(score_obj)

        # Return overall score (or pick main one like "correctness")
        return scores_dict.get("correctness", {}).get("score") or self.default_value