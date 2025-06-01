import re
from co_ai.constants import REVIEW
from co_ai.models import ScoreORM
from co_ai.scoring.base_score import BaseScore


class ReviewScore(BaseScore):
    name = REVIEW
    default_value = 0.0

    def __init__(self, cfg, memory, logger=None):
        super().__init__(cfg, memory, logger)
        self.scores = {}

    def compute(self, hypothesis: dict, context: dict) -> float:
        review_text = hypothesis.get(REVIEW)
        if not review_text:
            return self.default_value

        try:
            self.scores = self._extract_scores(review_text)
            if not self.scores:
                return self.default_value

            # Cache full dict for later use
            hypothesis["review_scores"] = self.scores

            # Save each dimension as a separate score
            for dimension, data in self.scores.items():
                if dimension == "overall":
                    score_type = "overall"
                    score = data.get("score", 0.0)
                    rationale = data.get("summary", "")
                else:
                    score_type = dimension
                    score = data.get("score", 0.0)
                    rationale = data.get("rationale", "")

                self._store_score(hypothesis, context, score_type, score, rationale)

            return self.scores.get("overall", {}).get("score", self.default_value)

        except Exception as e:
            self.logger.log("ReviewScoreParseFailed", {
                "hypothesis_id": hypothesis.get("id"),
                "error": str(e),
                "review_snippet": review_text[:300],
            })
            return self.default_value

    def _extract_scores(self, text: str) -> dict:
        """
        Extract review scores from structured plain text review output.
        Expected fields:
          - correctness
          - originality
          - clarity
          - relevance
          - overall_score
          - summary
          - suggested_improvements (as list)
        """
        scores = {}

        def extract_int(tag):
            match = re.search(rf"{tag}:\s*(\d+)", text)
            return int(match.group(1)) if match else None

        def extract_text(tag):
            match = re.search(rf"{tag}:\s*(.*?)(?=\n\w+:|\noverall_score:|$)", text, re.DOTALL)
            return match.group(1).strip() if match else ""

        for dim in ["correctness", "originality", "clarity", "relevance"]:
            score = extract_int(dim)
            rationale = extract_text(f"{dim}_rationale")
            if score is not None:
                scores[dim] = {"score": score, "rationale": rationale}

        overall = extract_int("overall_score")
        summary = extract_text("summary")
        improvements = re.findall(r"-\s*(.+)", text.split("suggested_improvements:")[-1]) if "suggested_improvements" in text else []

        scores["overall"] = {
            "score": overall if overall is not None else self.default_value,
            "summary": summary,
            "suggested_improvements": improvements,
        }

        return scores

    def _store_score(self, hypothesis: dict, context: dict, score_type: str, score: float, rationale: str):
        score_obj = ScoreORM(
            goal_id=hypothesis.get("goal_id"),
            hypothesis_id=hypothesis.get("id"),
            agent_name=self.agent_name,
            model_name=self.model_name,
            evaluator_name="ReviewScore",
            score_type=score_type,
            score=score,
            rationale=rationale,
            pipeline_run_id=context.get("pipeline_run_id"),
            metadata={"source": "structured_review"},
        )
        self.memory.scores.insert(score_obj)

    def get_score(self, hypothesis: dict, context: dict) -> float:
        return self.compute(hypothesis, context)
