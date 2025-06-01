from collections import defaultdict
import json
import math
from sqlalchemy.orm import Session
from co_ai.models import RuleApplicationORM, ScoreORM, ScoreRuleLinkORM
from tabulate import tabulate


class RuleEffectAnalyzer:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger

    def _compute_stats(self, scores: list[float]) -> dict:
        if not scores:
            return {}

        avg = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        std = math.sqrt(sum((x - avg) ** 2 for x in scores) / len(scores))
        success_rate = len([s for s in scores if s >= 50]) / len(scores)

        return {
            "avg_score": avg,
            "count": len(scores),
            "min": min_score,
            "max": max_score,
            "std": std,
            "success_rate": success_rate,
        }

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
            rule_summary = self._compute_stats(scores)
            results[rule_id] = {
                **rule_summary,
                "by_params": {},
            }

            print(f"\n📘 Rule {rule_id} Summary:")
            print(tabulate([
                ["Average Score", f"{rule_summary['avg_score']:.2f}"],
                ["Count", rule_summary["count"]],
                ["Min / Max", f"{rule_summary['min']} / {rule_summary['max']}"],
                ["Std Dev", f"{rule_summary['std']:.2f}"],
                ["Success Rate ≥50", f"{rule_summary['success_rate']:.2%}"],
            ], tablefmt="fancy_grid"))

            for param_key, score_list in param_scores[rule_id].items():
                param_summary = self._compute_stats(score_list)
                results[rule_id]["by_params"][param_key] = param_summary

                print(f"\n    🔧 Param Config: {param_key}")
                print(tabulate([
                    ["Average Score", f"{param_summary['avg_score']:.2f}"],
                    ["Count", param_summary["count"]],
                    ["Min / Max", f"{param_summary['min']} / {param_summary['max']}"],
                    ["Std Dev", f"{param_summary['std']:.2f}"],
                    ["Success Rate ≥50", f"{param_summary['success_rate']:.2%}"],
                ], tablefmt="rounded_outline"))
        return results
