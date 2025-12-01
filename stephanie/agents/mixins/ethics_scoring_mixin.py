# stephanie/agents/mixins/ethics_scoring_mixin.py
from __future__ import annotations

from textwrap import wrap

from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.scorable import Scorable, ScorableFactory, ScorableType


class EthicsScoringMixin:
    def score_ethics(self, context: dict, doc: dict) -> dict:
        context = context or {}
        context["ethics_score"] = doc

        if not hasattr(self, "call_llm"):
            raise AttributeError(
                "Agent must implement `call_llm(prompt, context)`"
            )

        evaluator: PaperScoreEvaluator = PaperScoreEvaluator.from_file(
            filepath=self.cfg.get(
                "score_config", "config/scoring/ethics.yaml"
            ),
            prompt_loader=self.prompt_loader,
            cfg=self.cfg,
            logger=self.logger,
            memory=self.memory,
        )

        return evaluator.evaluate(
            document=doc,
            context=context,
            llm_fn=self.call_llm,
            text_to_evaluate="summary",
        )


class PaperScoreEvaluator:
    def evaluate(
        self,
        context: dict,
        document: dict,
        llm_fn=None,
        text_to_evaluate: str = "text",
        max_tokens: int = 5000
    ) -> dict:
        text = document.get(text_to_evaluate, "")
        chunks = self.chunk_text(text, max_tokens=max_tokens)  # Adjust token limit as needed
        scores_accumulator = []

        for chunk in chunks:
            temp_paper = document.copy()
            temp_paper["text"] = chunk
            context["document"] = document
            chunk_context = {**context, "paper_score": temp_paper}

            scorable = Scorable(
                id=document.get("id"),
                text=chunk,
                target_type=ScorableType.CHUNK,
            )
            result = super().evaluate(chunk_context, scorable)

            scores_accumulator.append(result)

        dicts = [bundle.to_dict() for bundle in scores_accumulator]

        # Aggregate across chunks
        final_scores = self.aggregate_scores(dicts)
        bundle = ScoreBundle.from_dict(final_scores)
        scorable = ScorableFactory.from_dict(document, ScorableType.DOCUMENT)
        eval_id = self.memory.evaluations.save_bundle(
            bundle=bundle,
            scorable=scorable,
            context=context,
            cfg=self.cfg,
            agent_name=self.name,
            source="paper_score_evaluator", 
            model_name="ensemble", 
            evaluator_name=str(self.enabled_scorers)
        )
        self.logger.log("EvaluationSaved", {"id": eval_id})

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

        for dim in self.dimension_specs:
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
                "source": "paper_score_evaluator",
                "dimension": name
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
