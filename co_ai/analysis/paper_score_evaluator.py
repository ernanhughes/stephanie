# co_ai/scoring/paper_score_evaluator.py
from textwrap import wrap

from co_ai.analysis.score_evaluator import ScoreEvaluator
from co_ai.models import EvaluationORM, ScoreORM


class PaperScoreEvaluator(ScoreEvaluator):
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
        self.save_score_to_memory(final_scores, document, context)
        return final_scores

    def save_score_to_memory(self, results, paper, context):
        """
        Save paper_score evaluation scores to the database. Optionally link to a goal or pipeline run.
        """
        goal = context.get("goal")
        pipeline_run_id = context.get("pipeline_run_id")
        document_id = paper.get("id")  # assuming the paper_score is stored in DB already

        weighted_score = sum(
            s["score"] * s.get("weight", 1.0) for s in results.values()
        ) / max(sum(s.get("weight", 1.0) for s in results.values()), 1.0)

        scores_json = {
            "stage": self.cfg.get("stage", "review"),
            "dimensions": results,
            "final_score": round(weighted_score, 2),
        }

        # ✅ 1. Create EvaluationORM entry (new: paper_id field)
        eval_orm = EvaluationORM(
            goal_id=goal.get("id") if goal else None,
            pipeline_run_id=pipeline_run_id,
            document_id=document_id,  # custom field, add if not present
            agent_name=self.cfg.get("name"),
            model_name=self.cfg.get("model", {}).get("name"),
            evaluator_name=self.cfg.get("evaluator", "PaperScoreEvaluator"),
            strategy=self.cfg.get("strategy"),
            reasoning_strategy=self.cfg.get("reasoning_strategy"),
            scores=scores_json,
            extra_data={"source": "PaperScoreEvaluator"},
        )
        self.memory.session.add(eval_orm)
        self.memory.session.flush()  # Get eval_orm.id

        # ✅ 2. Create individual ScoreORM entries per dimension
        for dimension_name, result in results.items():
            score = ScoreORM(
                evaluation_id=eval_orm.id,
                dimension=dimension_name,
                score=result["score"],
                weight=result["weight"],
                rationale=result["rationale"],
            )
            self.memory.session.add(score)

        self.memory.session.commit()

        # ✅ 3. Log and optionally print summary
        self.logger.log(
            "PaperScoreSavedToMemory",
            {
                "document_id": document_id,
                "goal_id": goal.get("id") if goal else None,
                "scores": scores_json,
            },
        )

        self.display_results(results, weighted_score)

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
