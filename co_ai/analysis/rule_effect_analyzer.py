from collections import defaultdict
import json
from sqlalchemy.orm import Session
from co_ai.models import RuleApplicationORM, ScoreORM, ScoreRuleLinkORM


class RuleEffectAnalyzer:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger

    def analyze(self) -> dict:
        """
        Analyze rule effectiveness by collecting all scores linked to rule applications.

        Returns:
            dict: rule_id → summary of performance metrics, broken down by param config.
        """
        rule_scores = defaultdict(list)
        param_scores = defaultdict(lambda: defaultdict(list))  # rule_id → param_json → scores

        links = self.session.query(ScoreRuleLinkORM).all()

        for link in links:
            score = self.session.get(ScoreORM, link.score_id)
            rule_app = self.session.get(RuleApplicationORM, link.rule_application_id)

            if not score or not rule_app or score.score is None:
                if self.logger:
                    self.logger.log("SkipScoreLink", {
                        "reason": "missing score or rule_app or score is None",
                        "score_id": getattr(link, "score_id", None),
                        "rule_application_id": getattr(link, "rule_application_id", None),
                    })
                continue

            rule_id = rule_app.rule_id
            rule_scores[rule_id].append(score.score)

            # Normalize stage_details as sorted JSON
            try:
                param_key = json.dumps(rule_app.stage_details or {}, sort_keys=True)
            except Exception as e:
                param_key = "{}"
                if self.logger:
                    self.logger.log("StageDetailsParseError", {
                        "error": str(e),
                        "raw_value": str(rule_app.stage_details),
                    })

            param_scores[rule_id][param_key].append(score.score)

        # Build summary output
        results = {}
        for rule_id, scores in rule_scores.items():
            results[rule_id] = {
                "avg_score": sum(scores) / len(scores),
                "count": len(scores),
                "min": min(scores),
                "max": max(scores),
                "by_params": {},
            }

            for param_key, param_scores_list in param_scores[rule_id].items():
                results[rule_id]["by_params"][param_key] = {
                    "avg_score": sum(param_scores_list) / len(param_scores_list),
                    "count": len(param_scores_list),
                    "min": min(param_scores_list),
                    "max": max(param_scores_list),
                }

        return results
