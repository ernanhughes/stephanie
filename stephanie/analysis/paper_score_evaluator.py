# stephanie/analysis/paper_score_evaluator.py
from textwrap import wrap

from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.scoring_manager import ScoringManager


class PaperScoreEvaluator(ScoringManager):
    def evaluate(
        self,
        context: dict,
        document: dict,
        llm_fn=None,
        text_to_evaluate: str = "content",
    ) -> dict:
        text = document.get(text_to_evaluate, "")
        chunks = self.chunk_text(text, max_tokens=1000)  # Adjust token limit as needed
        scores_accumulator = []

        for chunk in chunks:
            temp_paper = document.copy()
            temp_paper["text"] = chunk
            context["document"] = document
            chunk_context = {**context, "paper_score": temp_paper}

            scorable = Scorable(
                id=document.get("id"),
                text=chunk,
                target_type=TargetType.CHUNK,
            )
            result = super().evaluate(chunk_context, scorable)

            scores_accumulator.append(result)

        dicts = [bundle.to_dict() for bundle in scores_accumulator]

        # Aggregate across chunks
        final_scores = self.aggregate_scores(dicts)
        final_bundle = ScoreBundle.from_dict(final_scores)
        ScoringManager.save_document_score_to_memory(
            final_bundle, document, context, self.cfg, self.memory, self.logger
        )
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
        Each input in `chunk_results` is a dict mapping dimension -> {score, rationale, weight}.
        """
        combined = {}

        for dim in self.dimensions:
            name = dim["name"]
            weight = dim.get("weight", 1.0)

            scores = []
            rationales = []

            for result in chunk_results:
                data = result.get(name)
                if data:
                    try:
                        score = float(data.get("score", 0))
                        scores.append(score)
                    except (TypeError, ValueError):
                        continue

                    rationale = data.get("rationale")
                    if rationale:
                        rationales.append(rationale.strip())

            if scores:
                avg_score = sum(scores) / len(scores)
            else:
                avg_score = 0.0  # or None, depending on how you want to handle it

            combined[name] = {
                "score": round(avg_score, 4),
                "rationale": "\n---\n".join(rationales[:3]),  # cap rationale size
                "weight": weight,
            }

        return combined


    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        results = []
        for document in documents:
            self.logger.log("ScoringPaper", {"title": document.get("title")})
            score_result = self.score_paper(document, context=context)
            results.append(
                {
                    "title": document.get("title"),
                    "scores": score_result,
                }
            )
        context[self.output_key] = results
        return context