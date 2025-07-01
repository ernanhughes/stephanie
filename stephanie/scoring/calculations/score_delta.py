class ScoreDeltaCalculator:
    def __init__(self, cfg: dict, memory, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

    def log_score_delta(self, hypothesis_id, new_score, goal_id=None):
        prev = self.memory.evaluations.get_latest_score(hypothesis_id, stage=self.cfg.get("name"))
        if prev is not None:
            delta = round(new_score - prev, 2)
            if self.logger:
                self.logger.log("ScoreDelta", {
                    "goal_id": goal_id,
                    "hypothesis_id": hypothesis_id,
                    "prev_score": prev,
                    "new_score": new_score,
                    "delta": delta,
                    "stage": self.cfg.get("name")
                })
            return delta
        return None
