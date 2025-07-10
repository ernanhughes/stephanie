import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy.orm import Session

from stephanie.models import BeliefORM, BeliefTuneLogORM, WorldviewORM


class WorldviewAuditAgent:
    def __init__(self, db: Session, logger=None):
        self.db = db
        self.logger = logger

    def get_belief_tuning_log(self, worldview_id: int):
        beliefs = self.db.query(BeliefORM).filter_by(worldview_id=worldview_id).all()
        log_entries = []
        for belief in beliefs:
            logs = self.db.query(BeliefTuneLogORM).filter_by(belief_id=belief.id).all()
            for log in logs:
                log_entries.append({
                    "belief_id": belief.id,
                    "title": belief.title,
                    "old_score": log.old_score,
                    "new_score": log.new_score,
                    "source": log.source,
                    "rationale": log.rationale,
                    "tuned_at": log.tuned_at
                })
        return log_entries

    def visualize_score_drift(self, belief_id: int):
        logs = self.db.query(BeliefTuneLogORM).filter_by(belief_id=belief_id).order_by(BeliefTuneLogORM.tuned_at).all()
        times = [log.tuned_at for log in logs]
        scores = [log.new_score for log in logs]

        plt.figure()
        plt.plot(times, scores, marker='o')
        plt.title(f"Belief {belief_id} Score Drift")
        plt.xlabel("Time")
        plt.ylabel("Score")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def source_influence_summary(self, worldview_id: int):
        logs = self.get_belief_tuning_log(worldview_id)
        df = pd.DataFrame(logs)
        return df.groupby("source")["belief_id"].count().sort_values(ascending=False)

    def get_recent_merges(self, worldview_id: int):
        # Requires merge logs or versioning system to be in place
        pass

    def flag_suspect_tuning(self, worldview_id: int, threshold: float = 0.5):
        logs = self.get_belief_tuning_log(worldview_id)
        df = pd.DataFrame(logs)
        df["score_change"] = (df["new_score"] - df["old_score"]).abs()
        flagged = df[df["score_change"] > threshold]
        return flagged
