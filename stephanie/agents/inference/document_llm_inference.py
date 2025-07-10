# stephanie/agents/knowledge/document_reward_scorer.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.evaluation import EvaluationORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.scorable_factory import ScorableFactory
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.scoring_engine import ScoringEngine
from stephanie.models.score import ScoreORM

DEFAULT_DIMENSIONS = [
    "alignment",
    "implementability",
    "clarity",
    "relevance",
    "novelty",
]


class DocumentLLMInferenceAgent(ScoringMixin, BaseAgent):
    """
    Scores document sections or full documents to assess reward value
    using configured reward model (e.g., SVM-based or regression-based).
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", DEFAULT_DIMENSIONS)
        self.scoring_engine = ScoringEngine(
            cfg=self.cfg,
            memory=self.memory,
            prompt_loader=self.prompt_loader,
            logger=self.logger,
            call_llm=self.call_llm,
        )

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        results = []
        for document in documents:
            doc_id = document["id"]

            saved_scores = self.get_scores_by_document_id(doc_id)
            if saved_scores:
                self.logger.log(
                    "DocumentScoresAlreadyExist",
                    {"document_id": doc_id, "num_scores": len(saved_scores)},
                )
                continue

            scorable = ScorableFactory.from_dict(document, TargetType.DOCUMENT)
            result = self.scoring_engine.score_item(scorable, context, "document")
            results.append(result)
            self.logger.log(
                "DocumentScored",
                {
                    "document_id": doc_id,
                    "title": document.get("title"),
                },
            )
            ScoringManager.save_score_to_memory(result, scorable, context, self.cfg, self.memory, self.logger, source="llm")
        self.logger.log("DocumentLLMInferenceCompleted", {"total_documents_scored": len(results)})
        # Store results in context for further processing
        context[self.output_key] = results
        return context

    def get_scores_by_document_id(self, document_id: int) -> list[ScoreORM]:
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
