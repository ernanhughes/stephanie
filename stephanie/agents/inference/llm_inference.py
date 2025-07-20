# stephanie/agents/inference/document_llm_inference.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scoring_engine import ScoringEngine
from stephanie.scoring.scoring_manager import ScoringManager

DEFAULT_DIMENSIONS = [
    "alignment",
    "implementability",
    "clarity",
    "relevance",
    "novelty",
]


class LLMInferenceAgent(ScoringMixin, BaseAgent):
    """
    Scores document sections or full documents to assess reward value
    using configured reward model (e.g., SVM-based or regression-based).
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "llm"
        self.evaluator = "llm"
        self.force_rescore = cfg.get("force_rescore", False)

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
            if saved_scores and not self.force_rescore:
                self.logger.log(
                    "DocumentScoresAlreadyExist",
                    {"document_id": doc_id, "num_scores": len(saved_scores)},
                )
                continue

            scorable = ScorableFactory.from_dict(document, TargetType.DOCUMENT)
            result = self.scoring_engine.score_item(
                scorable, context, "document"
            )
            results.append(result.to_dict())
            self.logger.log(
                "DocumentScored",
                {
                    "document_id": doc_id,
                    "title": document.get("title"),
                },
            )
            ScoringManager.save_score_to_memory(
                result,
                scorable,
                context,
                self.cfg,
                self.memory,
                self.logger,
                source="llm",
            )
        self.logger.log(
            "DocumentLLMInferenceCompleted",
            {"total_documents_scored": len(results)},
        )
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

    def score(self, context: dict, scorable: Scorable) -> dict:
        """
        Score a single text input using the LLM scoring engine across all configured dimensions.
        Returns a dictionary of {dimension: score}.
        """
        result = self.scoring_engine.score_item(
            scorable, context, "document"
        )
        ScoringManager.save_score_to_memory(
            result,
            scorable,
            context,
            self.cfg,
            self.memory,
            self.logger,
            source="llm",
        )

        return result
