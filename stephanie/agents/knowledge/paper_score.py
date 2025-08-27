# stephanie/agents/knowledge/paper_score.py

from collections import defaultdict

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.paper_scoring_mixin import PaperScoringMixin
from stephanie.models import EvaluationORM, ScoreORM
from stephanie.scoring.scorable_factory import TargetType


class PaperScoreAgent(BaseAgent, PaperScoringMixin):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.force_rescore = cfg.get("force_rescore", True)

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        results = []

        self.report({
            "event": "start",
            "step": "PaperScoring",
            "details": f"Scoring {len(documents)} documents",
        })

        for document in documents:
            doc_id_str = str(document["id"])
            title = document.get("title", "Untitled")

            existing_scores = self.scores_exist_for_document(doc_id_str)
            if existing_scores and not self.force_rescore:
                scores = self.get_scores_by_document_id(doc_id_str)
                aggregated = self.aggregate_scores_by_dimension(scores)

                self.report({
                    "event": "skipped_existing",
                    "step": "PaperScoring",
                    "doc_id": doc_id_str,
                    "title": title[:80],
                    "scores": aggregated,
                })

                results.append({"title": title, "scores": aggregated})
                continue

            # Fresh scoring
            self.report({
                "event": "scoring_started",
                "step": "PaperScoring",
                "doc_id": doc_id_str,
                "title": title[:80],
            })

            score_result = self.score_paper(document, context=context)

            self.report({
                "event": "scored",
                "step": "PaperScoring",
                "doc_id": doc_id_str,
                "title": title[:80],
                "scores": score_result,
            })

            results.append({"title": title, "scores": score_result})

        context[self.output_key] = results

        self.report({
            "event": "end",
            "step": "PaperScoring",
            "details": f"Completed scoring {len(results)} documents",
        })

        return context

    def assign_domains_to_document(self, document):
        """
        Classifies the document text into one or more domains,
        and stores results in the document_domains table.
        """
        # Skip if already has domains
        if self.memory.document_domains.get_domains(document.id):
            return

        text = document.text or ""
        results = self.domain_classifier.classify(text)

        for domain, score in results:
            self.memory.document_domains.insert(
                {
                    "document_id": document.id,
                    "domain": domain,
                    "score": score,
                }
            )

            self.logger.log(
                "DomainAssignedAgent",
                {
                    "title": document.title[:60] if document.title else "",
                    "domain": domain,
                    "score": score,
                },
            )

    def get_scores_by_document_id(self, document_id: str) -> list[ScoreORM]:
        evaluations = (
            self.memory.session.query(EvaluationORM)
            .filter_by(target_type=TargetType.DOCUMENT, target_id=document_id)
            .all()
        )

        scores = []
        for evaluation in evaluations:
            scores.extend(
                self.memory.session.query(ScoreORM)
                .filter_by(evaluation_id=evaluation.id)
                .all()
            )
        return scores

    def scores_exist_for_document(self, document_id: str) -> bool:
        return self.memory.session.query(ScoreORM.id).join(
            EvaluationORM, ScoreORM.evaluation_id == EvaluationORM.id
        ).filter(
            EvaluationORM.target_type == TargetType.DOCUMENT,
            EvaluationORM.target_id == document_id
        ).first() is not None

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
