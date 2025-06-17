# co_ai/agents/knowledge/paper_scoring_adapter.py

from collections import defaultdict

from co_ai.agents.base_agent import BaseAgent
from co_ai.agents.mixins.paper_scoring_mixin import PaperScoringMixin
from co_ai.models import EvaluationORM, ScoreORM


class PaperScoreAgent(BaseAgent, PaperScoringMixin):

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.force_rescore = cfg.get("force_rescore", False)

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        results = []
        for document in documents:
            doc_id = document["id"]
            existing_scores = self.get_scores_by_document_id(doc_id)
            if existing_scores and not self.force_rescore:
                self.logger.log(
                    "ScoreSkipped",
                    {"doc_id": doc_id, "score": existing_scores},
                )
                results.append(
                    {
                        "title": document.get("title"),
                        "scores": self.aggregate_scores_by_dimension(existing_scores)
                    }
                )
                continue

            self.logger.log("ScoringPaper", {"title": document.get("title")})
            score_result = self.score_paper(document, context=context)
            results.append({
                "title": document.get("title"),
                "scores": score_result,
            })
        context[self.output_key] = results
        return context

    def get_scores_by_document_id (self, document_id: int) -> list[ScoreORM]:
        evaluations = self.memory.session.query(EvaluationORM).filter_by(document_id=document_id).all()
        scores = []
        for evaluation in evaluations:
            scores.extend(
                self.memory.session.query(ScoreORM)
                .filter_by(evaluation_id=evaluation.id)
                .all()
            )
        return scores

    def aggregate_scores_by_dimension(self, scores: list[ScoreORM]) -> dict:
        dimension_totals = defaultdict(list)

        for score_obj in scores:
            if score_obj.score != 0:  # Ignore zero (garbage) scores
                dimension_totals[score_obj.dimension].append(score_obj.score)

        # Average non-zero scores per dimension
        return {
            dim: round(sum(vals) / len(vals), 4)
            for dim, vals in dimension_totals.items()
            if vals  # Only include dimensions that had non-zero values
        }