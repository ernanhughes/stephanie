# co_ai/scoring/paper_score_evaluator.py
from textwrap import wrap

from co_ai.models import EvaluationORM, ScoreORM
from co_ai.scoring.scoring_manager import ScoringManager


class PaperScoreEvaluator(ScoringManager):
    def evaluate(self, document: dict, context: dict = {}, llm_fn=None):
        text = document.get("content", "")
        chunks = self.chunk_text(text, max_tokens=1000)  # Adjust token limit as needed
        scores_accumulator = []

        for chunk in chunks:
            temp_paper = document.copy()
            temp_paper["text"] = chunk
            context["document"] = document
            chunk_context = {**context, "paper_score": temp_paper}

            if self.output_format == "cor":
                result = super()._evaluate_cor(temp_paper, chunk_context, llm_fn)
            else:
                result = super()._evaluate_simple(temp_paper, chunk_context, llm_fn)

            scores_accumulator.append(result)

        # Aggregate across chunks
        final_scores = self.aggregate_scores(scores_accumulator)
        ScoringManager.save_score_to_memory(final_scores, document, context)
        return final_scores


    def chunk_text(self, text: str, max_tokens: int = 1000) -> list[str]:
        """
        Splits the text into chunks based on token count approximation.
        """
        # Approximate 1 token ≈ 4 characters for English
        max_chars = max_tokens * 4
        return wrap(text, width=max_chars)

    def aggregate_scores(self, chunk_results: list[dict]) -> dict:
        """
        Average scores and concatenate rationales across chunks.
        """
        combined = {}
        count = len(chunk_results)

        for dim in self.dimensions:
            name = dim["name"]
            scores = [r[name]["score"] for r in chunk_results if name in r]
            rationales = [r[name]["rationale"] for r in chunk_results if name in r]

            avg_score = sum(scores) / max(len(scores), 1)
            combined[name] = {
                "score": avg_score,
                "rationale": "\n---\n".join(rationales[:3]),  # limit rationale bloat
                "weight": dim["weight"],
            }

        return combined
