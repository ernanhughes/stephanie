# stephanie/agents/knowledge/document_reward_scorer.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.ebt_scorer import EBTScorer
from stephanie.scoring.mrq_scorer import MRQScorer
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.svm_scorer import SVMScorer


class DocumentRewardScorerAgent(BaseAgent):
    """
    Scores document sections or full documents to assess reward value
    using configured reward model (e.g., SVM-based or regression-based).
    """

    def __init__(self, cfg, memory=None, logger=None, scorer: SVMScorer = None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", [])
        

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        results = []

        svm_scorer = SVMScorer(self.cfg, memory=self.memory, logger=self.logger)
        mrq_scorer = MRQScorer(self.cfg, memory=self.memory, logger=self.logger)
        sicql_scorer = SICQLScorer(self.cfg, memory=self.memory, logger=self.logger)  
        ebt_scorer = EBTScorer(self.cfg, memory=self.memory, logger=self.logger)

        scorers = [sicql_scorer, svm_scorer, mrq_scorer, ebt_scorer]

        for doc in documents:

            doc_id = doc["id"]
            goal = context.get("goal", "")
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)

            for scorer in scorers:
                score_bundle: ScoreBundle = scorer.score(
                    goal=goal,
                    scorable=scorable,
                    dimensions=self.dimensions,
                )

                if self.logger:
                    self.logger.log(
                        "DocumentScored",
                        {
                            "document_id": doc_id,
                            "title": doc.get("title"),
                            "scores": score_bundle.to_dict(),
                        },
                    )

                ScoringManager.save_score_to_memory(
                    score_bundle,
                    scorable,
                    context,
                    self.cfg,
                    self.memory,
                    self.logger,
                    source=scorer.model_type,
                    model_name=scorer.get_model_name(),
                )


            results.append(
                {
                    "document_id": doc_id,
                    "title": doc.get("title"),
                    "scores": score_bundle.to_dict(),
                }
            )

        context[self.output_key] = results
        return context

