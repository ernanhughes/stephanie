# stephanie/agents/knowledge/document_reward_scorer.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.svm_scorer import SVMScorer
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.models.score import ScoreORM
from stephanie.models.evaluation import EvaluationORM


class DocumentRewardScorerAgent(BaseAgent):
    """
    Scores document sections or full documents to assess reward value
    using configured reward model (e.g., SVM-based or regression-based).
    """

    def __init__(self, cfg, memory=None, logger=None, scorer: SVMScorer = None):
        super().__init__(cfg, memory, logger)
        self.scorer = scorer or SVMScorer(cfg, memory=memory, logger=logger)

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        results = []

        for doc in documents:
            doc_id = doc["id"]
            goal = {"goal_text": context.get("goal", "")}
            hypothesis = {"text": doc.get("summary") or doc.get("text", "")}

            score_bundle: ScoreBundle = self.scorer.score(
                goal=goal,
                hypothesis=hypothesis,
                dimensions=["alignment", "actionability", "clarity"]
            )

            if self.logger:
                self.logger.log("DocumentScored", {
                    "document_id": doc_id,
                    "title": doc.get("title"),
                    "scores": score_bundle.to_dict()
                })

            # Persist results
            evaluation_id = self._store_evaluation(doc_id, context)
            self._store_scores(score_bundle, evaluation_id)

            results.append({
                "document_id": doc_id,
                "title": doc.get("title"),
                "scores": score_bundle.to_dict()
            })

        context[self.output_key] = results
        return context

    def _store_evaluation(self, document_id, context) -> int:
        evaluation = EvaluationORM(
            document_id=document_id,
            goal_text=context.get("goal", ""),
            metadata={"source": "reward_scorer"}
        )
        self.memory.session.add(evaluation)
        self.memory.session.commit()
        return evaluation.id

    def _store_scores(self, bundle: ScoreBundle, evaluation_id: int):
        for score_orm in bundle.to_orm(evaluation_id=evaluation_id):
            self.memory.session.add(score_orm)
        self.memory.session.commit()
