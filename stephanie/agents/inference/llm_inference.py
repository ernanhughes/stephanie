# stephanie/agents/inference/document_llm_inference.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scoring_engine import ScoringEngine
from stephanie.scoring.scoring_manager import ScoringManager
from tqdm import tqdm


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

        self.dimensions = cfg.get("dimensions", [])
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

        # Add progress bar over documents
        for document in tqdm(documents, desc="🔍 LLM Scoring Progress", unit="doc"):
            doc_id = document["id"]
            scorable = ScorableFactory.from_dict(document, TargetType.DOCUMENT)

            score_count = self.get_score_count_by_document_id(scorable.id)
            if score_count > 0 and not self.force_rescore:
                self.logger.log(
                    "DocumentScoresAlreadyExist",
                    {"document_id": doc_id, "num_scores": score_count},
                )
                continue

            result = self.scoring_engine.score_item(scorable, context, "document")
            results.append(result.to_dict())

            self.logger.log("DocumentScored", {
                "document_id": doc_id,
                "title": document.get("title"),
            })

            ScoringManager.save_score_to_memory(
                result,
                scorable,
                context,
                self.cfg,
                self.memory,
                self.logger,
                source="llm",
            )

        self.logger.log("DocumentLLMInferenceCompleted", {
            "total_documents_scored": len(results),
        })

        context[self.output_key] = results
        return context

    def get_scores_by_document_id(self, scorable_id: str) -> list[ScoreORM]:
        evaluations = (
            self.memory.session.query(EvaluationORM)
            .filter_by(target_type=TargetType.DOCUMENT, target_id=scorable_id)
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

    def get_score_count_by_document_id(self, scorable_id: str) -> int:
        """
        Fast check: how many scores exist for this document.
        Uses a single join query.
        """
        count = (
            self.memory.session.query(ScoreORM)
            .join(EvaluationORM, ScoreORM.evaluation_id == EvaluationORM.id)
            .filter(
                EvaluationORM.target_type == "document",
                EvaluationORM.target_id == str(scorable_id),
                EvaluationORM.source == "llm"
            )
            .count()
        )
        return count
