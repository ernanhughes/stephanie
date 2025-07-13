# stephanie/scoring/calculations/score_delta.py
class ScoreDeltaCalculator:
    def __init__(self, cfg: dict, memory, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

    def log_score_delta(self, scorable, new_score, goal_id=None):
        prev = self.memory.evaluations.get_latest_score(
            scorable, agent_name=self.cfg.get("name")
        )
        if prev is not None:
            delta = round(new_score - prev, 2)
            if self.logger:
                self.logger.log(
                    "ScoreDelta",
                    {
                        "delta": delta,
                        "id": scorable.id,
                        "target_type": scorable.target_type,
                        "text": scorable.text[:60],
                        "goal_id": goal_id,
                        "prev_score": prev,
                        "new_score": new_score,
                        "stage": self.cfg.get("name"),
                    },
                )
            return delta
        return None
