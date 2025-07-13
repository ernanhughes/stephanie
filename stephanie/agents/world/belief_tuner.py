# stephanie/agents/world/belief_tuner.py
from datetime import datetime

from sqlalchemy.orm import Session

from stephanie.models.belief import BeliefORM


class BeliefTunerAgent:
    def __init__(self, db: Session, logger=None):
        self.db = db
        self.logger = logger

    def tune_belief(
        self,
        belief_id: int,
        delta: float,
        source: str,
        rationale: str = None,
        override_score: float = None,
    ):
        """Adjust or override belief trust score"""
        belief = self.db.query(BeliefORM).get(belief_id)
        if not belief:
            return None

        old_score = belief.score or 0.5

        if override_score is not None:
            belief.score = override_score
        else:
            belief.score = max(0.0, min(1.0, old_score + delta))

        belief.last_tuned = datetime.utcnow()
        belief.last_tune_source = source
        belief.last_tune_rationale = rationale or "Tuned via agent"

        self.db.commit()

        self.logger.log(
            "BeliefTuned",
            {
                "belief_id": belief.id,
                "old_score": old_score,
                "new_score": belief.score,
                "source": source,
                "rationale": rationale,
            },
        )

        return belief

    def tune_by_external_signal(self, belief_text: str, signal: dict):
        """Find belief by text match and tune it based on external input"""
        matches = (
            self.db.query(BeliefORM)
            .filter(BeliefORM.summary.ilike(f"%{belief_text}%"))
            .all()
        )
        for belief in matches:
            self.tune_belief(
                belief_id=belief.id,
                delta=signal.get("delta", -0.2),
                source=signal.get("source", "external"),
                rationale=signal.get("rationale"),
            )
