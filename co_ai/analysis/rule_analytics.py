from collections import defaultdict
from typing import Dict, List, Optional

from co_ai.models.rule_application import RuleApplicationORM
from co_ai.models.symbolic_rule import SymbolicRuleORM


class RuleAnalytics:
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger

    def get_score_summary(self, rule_id: int) -> dict:
        scores = (
            self.db.session.query(RuleApplicationORM.post_score)
            .filter(RuleApplicationORM.rule_id == rule_id)
            .filter(RuleApplicationORM.post_score != None)
            .all()
        )
        values = [s[0] for s in scores]
        if not values:
            return {"average": None, "count": 0}
        return {
            "average": sum(values) / len(values),
            "count": len(values),
            "min": min(values),
            "max": max(values),
        }

    def get_feedback_summary(self, rule_id: int) -> Dict[str, int]:
        results = (
            self.db.session.query(RuleApplicationORM.change_type)
            .filter(RuleApplicationORM.rule_id == rule_id)
            .all()
        )
        summary = defaultdict(int)
        for (label,) in results:
            if label:
                summary[label] += 1
        return dict(summary)

    def compute_rule_rank(
        self,
        score_avg: Optional[float],
        usage_count: int,
        feedback: Dict[str, int]
    ) -> float:
        """Compute a basic rule quality score. Can be replaced with DPO/MRQ later."""
        if score_avg is None:
            return -float("inf")
        bonus = feedback.get("good", 0)
        penalty = feedback.get("bad", 0) * 0.5
        return score_avg + bonus - penalty

    def analyze_all_rules(self) -> List[dict]:
        rules = self.db.symbolic_rules.get_all_rules()
        output = []
        for rule in rules:
            score_summary = self.get_score_summary(rule.id)
            feedback_summary = self.get_feedback_summary(rule.id)
            rank = self.compute_rule_rank(
                score_summary.get("average"), score_summary.get("count"), feedback_summary
            )
            result = {
                "rule_id": rule.id,
                "rule_text": rule.rule_text,
                "target": rule.target,
                "attributes": rule.attributes,
                "score_summary": score_summary,
                "feedback_summary": feedback_summary,
                "rank_score": rank,
            }
            if self.logger:
                self.logger.log("RuleAnalysis", result)
            output.append(result)
        return output
