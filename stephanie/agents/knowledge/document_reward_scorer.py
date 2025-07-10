# stephanie/agents/knowledge/document_reward_scorer.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.evaluation import EvaluationORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.svm_scorer import SVMScorer

DEFAULT_DIMENSIONS = ["alignment", "implementability", "clarity", "relevance"]


class DocumentRewardScorerAgent(BaseAgent):
    """
    Scores document sections or full documents to assess reward value
    using configured reward model (e.g., SVM-based or regression-based).
    """

    def __init__(self, cfg, memory=None, logger=None, scorer: SVMScorer = None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", DEFAULT_DIMENSIONS)
        self.scorer = scorer or SVMScorer(cfg, memory=memory, logger=logger, dimensions=self.dimensions)

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        results = []

        for doc in documents:
            doc_id = doc["id"]
            goal = context.get("goal", "")
            text = doc.get("summary") or doc.get("content", "")
            scorable = Scorable(text=text, target_type=TargetType.DOCUMENT, id=doc_id)

            score_bundle: ScoreBundle = self.scorer.score(
                goal=goal,
                scorable=scorable,
                dimensions=self.dimensions,
            )

            if self.logger:
                self.logger.log("DocumentScored", {
                    "document_id": doc_id,
                    "title": doc.get("title"),
                    "scores": score_bundle.to_dict()
                })

            # Persist results
            evaluation_id = self._store_evaluation(scorable, context)
            self._store_scores(score_bundle, evaluation_id)

            results.append({
                "document_id": doc_id,
                "title": doc.get("title"),
                "scores": score_bundle.to_dict()
            })

        context[self.output_key] = results
        return context

    def _store_evaluation(self, scorable, context) -> int:
        evaluation = EvaluationORM(
            target_id=scorable.id,
            target_type=scorable.target_type,
            goal_id=context.get("goal", {}).get("id"),
            metadata={"source": "reward_scorer"},
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            model_name=self.model_name,
            evaluator_name=self.scorer.name,
            strategy=self.strategy,
        )
        self.memory.session.add(evaluation)
        self.memory.session.commit()
        return evaluation.id

    def _store_scores(self, bundle: ScoreBundle, evaluation_id: int):
        for score_orm in bundle.to_orm(evaluation_id=evaluation_id):
            self.memory.session.add(score_orm)
        self.memory.session.commit()
