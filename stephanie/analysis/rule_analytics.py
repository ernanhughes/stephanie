from collections import defaultdict
from typing import Dict, List, Optional

from tabulate import tabulate

from stephanie.models.rule_application import RuleApplicationORM


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
        table_rows = []
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
            output.append(result)

            avg_score = score_summary.get("average") or 0.0
            score_count = score_summary.get("count") or 0
            pos_feedback = feedback_summary.get("positive", 0)
            neg_feedback = feedback_summary.get("negative", 0)

            # Prepare a table row summary for printout
            table_rows.append([
                rule.id,
                rule.target or "—",
                rule.rule_text[:30] + "…" if rule.rule_text and len(rule.rule_text) > 30 else rule.rule_text,
                f"{avg_score:.2f}",
                score_count,
                pos_feedback,
                neg_feedback,
                f"{rank:.2f}",
                ])
            output.append(result)

        # Print final table
        print("\n📋 Rule Analysis Summary:")
        headers = [
            "Rule ID",
            "Target",
            "Rule Text",
            "Avg Score",
            "Score Count",
            "👍 Feedback",
            "👎 Feedback",
            "Rank Score",
        ]
        print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))
        return output

    def analyze_rules_for_run(self, pipeline_run_id: str) -> List[dict]:
        """Analyze rules used in a specific pipeline run."""
        rule_apps = (
            self.db.session.query(RuleApplicationORM)
            .filter(RuleApplicationORM.pipeline_run_id == pipeline_run_id)
            .all()
        )

        rules_by_id = defaultdict(list)
        for app in rule_apps:
            rules_by_id[app.rule_id].append(app)

        output = []
        table_rows = []
        for rule_id, applications in rules_by_id.items():
            scores = [app.post_score for app in applications if app.post_score is not None]
            changes = [app.change_type for app in applications if app.change_type]

            feedback_summary = defaultdict(int)
            for label in changes:
                feedback_summary[label] += 1

            avg_score = sum(scores) / len(scores) if scores else None
            rank = self.compute_rule_rank(avg_score, len(scores), feedback_summary)

            rule = self.db.symbolic_rules.get_by_id(rule_id)
            if rule:
                result = {
                    "rule_id": rule_id,
                    "rule_text": rule.rule_text,
                    "target": rule.target,
                    "attributes": rule.attributes,
                    "score_summary": {
                        "average": avg_score,
                        "count": len(scores),
                        "min": min(scores) if scores else None,
                        "max": max(scores) if scores else None,
                    },
                    "feedback_summary": dict(feedback_summary),
                    "rank_score": rank,
                }
                output.append(result)
                table_rows.append([
                    rule_id,
                    rule.target or "—",
                    rule.rule_text[:30] + "…" if rule.rule_text and len(rule.rule_text) > 30 else rule.rule_text,
                    f"{avg_score:.2f}" if avg_score is not None else "—",
                    len(scores),
                    feedback_summary.get("positive", 0),
                    feedback_summary.get("negative", 0),
                    f"{rank:.2f}" if avg_score is not None else "—",
                ])

        print(f"\n📊 Rule Analysis for Pipeline Run: {pipeline_run_id}")
        headers = [
            "Rule ID", "Target", "Rule Text", "Avg Score", "Score Count",
            "👍 Feedback", "👎 Feedback", "Rank Score",
        ]
        print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))

        return output
