from datetime import datetime

from sqlalchemy.orm import Session

from stephanie.models.belief import BeliefORM
from stephanie.models.cartridge import CartridgeORM
from stephanie.models.goal import GoalORM
from stephanie.scoring.svm_scorer import SVMScorer
from stephanie.utils.embedding import EmbeddingManager
from stephanie.utils.summarizer import summarize_text


class BeliefIngestAgent:
    def __init__(self, db: Session, scorer: SVMScorer, embedding: EmbeddingManager, logger=None):
        self.db = db
        self.scorer = scorer
        self.embedding = embedding
        self.logger = logger

    def ingest_document(self, text: str, worldview_id: int, goal_id: int = None, source_uri: str = None):
        """
        Extracts belief(s) from a document and stores them.
        """
        # Step 1: Summarize key point(s)
        summary = summarize_text(text)

        # Step 2: Score belief utility and novelty
        goal_text = self._get_goal_text(goal_id)
        score_bundle = self.scorer.score(
            goal={"goal_text": goal_text},
            hypothesis={"text": summary},
            dimensions=["alignment", "novelty"]
        )
        alignment_score = score_bundle.results["alignment"].score
        novelty_score = score_bundle.results["novelty"].score

        # Step 3: Store belief in DB
        belief = BeliefORM(
            worldview_id=worldview_id,
            cartridge_id=None,  # optional: link to a CartridgeORM if available
            summary=summary,
            rationale="Auto-ingested from source document",
            utility_score=alignment_score,
            novelty_score=novelty_score,
            domain=self._infer_domain(summary),
            status="active",
            created_at=datetime.utcnow()
        )

        self.db.add(belief)
        self.db.commit()

        if self.logger:
            self.logger.log("BeliefIngested", {
                "summary": summary,
                "alignment": alignment_score,
                "novelty": novelty_score,
                "worldview_id": worldview_id
            })

        return belief

    def _get_goal_text(self, goal_id: int):
        if not goal_id:
            return "Understand and extend self-improving AI systems"
        goal = self.db.query(GoalORM).filter_by(id=goal_id).first()
        return goal.description if goal else ""

    def _infer_domain(self, text: str):
        # Placeholder: you could do zero-shot classification here
        return "ai.research.self_improvement"
