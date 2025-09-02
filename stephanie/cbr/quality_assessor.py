# stephanie/cbr/quality_assessor.py
from typing import Dict

class DefaultQualityAssessor:
    def __init__(self, cfg, memory, logger):
        self.cfg, self.memory, self.logger = cfg, memory, logger
        self.qw = cfg.get("quality_weights", {}) or {"mars":1.0,"hrm":0.5,"reward":2.0,"llm":0.25}

    def quality(self, mars_results: Dict, scores_payload: Dict) -> float:
        mars_agree = 0.0
        if mars_results:
            vals = [float(v.get("agreement_score", 0.0)) for v in mars_results.values()]
            mars_agree = (sum(vals)/len(vals)) if vals else 0.0
        hrm_score = 0.0; llm_grade = 0.0; task_reward = 0.0
        return (self.qw["mars"]*mars_agree + self.qw["hrm"]*hrm_score +
                self.qw["llm"]*llm_grade + self.qw["reward"]*task_reward)
