import json
import math
from collections import defaultdict
from typing import Optional

from sqlalchemy.orm import Session
from tabulate import tabulate

from co_ai.models import (EvaluationORM, EvaluationRuleLinkORM, PipelineRunORM,
                          RuleApplicationORM)


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

    def analyze(self, pipeline_run_id: int) -> dict:
        """
        Analyze rule effectiveness by collecting all scores linked to rule applications.

        Returns:
            dict: rule_id → summary of performance metrics, broken down by param config.
        """
        rule_scores = defaultdict(list)
        param_scores = defaultdict(lambda: defaultdict(list))  # rule_id → param_json → scores

        # Join ScoreRuleLinkORM with RuleApplicationORM to filter on pipeline_run_id
        links = (
            self.session.query(EvaluationRuleLinkORM)
            .join(RuleApplicationORM, RuleApplicationORM.id == EvaluationRuleLinkORM.rule_application_id)
            .filter(RuleApplicationORM.pipeline_run_id == pipeline_run_id)
            .all()
        )

        for link in links:
            score = self.session.get(EvaluationORM, link.score_id)
            rule_app = self.session.get(RuleApplicationORM, link.rule_application_id)

            if not score or not rule_app:
                if self.logger:
                    self.logger.log("SkipScoreLink", {
                        "reason": "missing score or rule_app or score is None",
                        "score_id": getattr(link, "score_id", None),
                        "rule_application_id": getattr(link, "rule_application_id", None),
                    })
                continue

            rule_id = rule_app.rule_id
            rule_scores[rule_id].append(score.evaluations.get("final_score", 0.0))

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


            param_scores[rule_id][param_key].append(score.evaluations.get("final_score", 0.0))

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

    def pipeline_run_scores(self, pipeline_run_id: Optional[int] = None, context: dict = None) -> None:
        """
        Generate a summary log showing all scores for a specific pipeline run.

        Args:
            pipeline_run_id (Optional[int]): ID of the pipeline run to inspect.
            context (dict): Optional context containing 'pipeline_run_id' as fallback.
        """
        if pipeline_run_id is None:
            if context and "pipeline_run_id" in context:
                pipeline_run_id = context["pipeline_run_id"]
            else:
                raise ValueError("No pipeline_run_id provided or found in context.")

        pipeline_run = self.session.get(PipelineRunORM, pipeline_run_id)
        if not pipeline_run:
            raise ValueError(f"No pipeline run found with ID {pipeline_run_id}")

        scores = (
            self.session.query(EvaluationORM)
            .filter(EvaluationORM.pipeline_run_id == pipeline_run_id)
            .all()
        )

        if not scores:
            if self.logger:
                self.logger.log(
                    "PipelineRunScoreSummary",
                    {
                        "pipeline_run_id": pipeline_run_id,
                        "total_scores": 0,
                        "message": "No scores found",
                    },
                )
            return

        table_rows = []
        for score in scores:
            rule_app_link = (
                self.session.query(EvaluationRuleLinkORM)
                .filter(EvaluationRuleLinkORM.score_id == score.id)
                .first()
            )
            rule_app = (
                self.session.get(RuleApplicationORM, rule_app_link.rule_application_id)
                if rule_app_link
                else None
            )

            row = [
                score.id,
                score.agent_name or "N/A",
                score.model_name or "N/A",
                score.evaluator_name or "N/A",
                score.scores,
                rule_app.rule_id if rule_app else "—",
                score.hypothesis_id or "—",
            ]
            table_rows.append(row)

        headers = [
            "Score ID",
            "Agent",
            "Model",
            "Evaluator",
            "Type",
            "Value",
            "Rule ID",
            "Hypothesis ID",
        ]

        # Print the table
        print(f"\n📊 Scores for Pipeline Run {pipeline_run_id}:")
        print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))

        if self.logger:
            self.logger.log("PipelineRunScoreSummary", {
                "pipeline_run_id": pipeline_run_id,
                "total_scores": len(scores)
            })
