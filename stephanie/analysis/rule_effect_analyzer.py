# stephanie/analysis/rule_effect_analyzer.py
import json
import math
from collections import defaultdict
from typing import Optional

from sqlalchemy.orm import Session
from tabulate import tabulate

from stephanie.models import (EvaluationORM, EvaluationRuleLinkORM,
                              PipelineRunORM, RuleApplicationORM)


class RuleEffectAnalyzer:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger

    def _compute_stats(self, values):
        if not values:
            return {}
        avg = sum(values) / len(values)
        std = math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))
        return {
            "avg": avg,
            "min": min(values),
            "max": max(values),
            "std": std,
            "count": len(values),
            "success_rate_≥50": len([v for v in values if v >= 50]) / len(values),
        }

    def get_scores_for_evaluation(self, evaluation_id):
        from stephanie.models.score import \
            ScoreORM  # local import to avoid circularity

        return (
            self.session.query(ScoreORM)
            .filter_by(evaluation_id=evaluation_id)
            .order_by(ScoreORM.dimension.asc())
            .all()
        )

    def analyze(self, pipeline_run_id: int) -> dict:
        """
        Analyze rule effectiveness by collecting all scores linked to rule applications.

        Returns:
            dict: rule_id → summary of performance metrics, broken down by param config.
        """
        rule_scores = defaultdict(list)
        param_scores = defaultdict(
            lambda: defaultdict(list)
        )  # rule_id → param_json → scores

        # Join ScoreRuleLinkORM with RuleApplicationORM to filter on pipeline_run_id
        links = (
            self.session.query(EvaluationRuleLinkORM)
            .join(
                RuleApplicationORM,
                RuleApplicationORM.id == EvaluationRuleLinkORM.rule_application_id,
            )
            .filter(RuleApplicationORM.pipeline_run_id == pipeline_run_id)
            .all()
        )

        for link in links:
            eval_id = link.evaluation_id
            rule_app = self.session.get(RuleApplicationORM, link.rule_application_id)
            if not eval_id or not rule_app:
                continue

            rule_id = rule_app.rule_id
            try:
                param_key = json.dumps(rule_app.stage_details or {}, sort_keys=True)
            except Exception:
                param_key = "{}"

            scores = self.get_scores_for_evaluation(eval_id)
            for score in scores:
                dim = score.dimension
                val = score.value
                rule_scores[rule_id][dim].append(val)
                param_scores[rule_id][param_key][dim].append(val)

        # Output
        for rule_id, dim_dict in rule_scores.items():
            print(f"\n📘 Rule {rule_id} Dimensional Summary:")
            table = []
            for dim, vals in dim_dict.items():
                stats = self._compute_stats(vals)
                table.append(
                    [
                        dim,
                        f"{stats['avg']:.2f}",
                        f"{stats['min']:.1f} / {stats['max']:.1f}",
                        f"{stats['std']:.2f}",
                        stats["count"],
                        f"{stats['success_rate_≥50']:.0%}",
                    ]
                )
            print(
                tabulate(
                    table,
                    headers=[
                        "Dimension",
                        "Avg",
                        "Min/Max",
                        "Std",
                        "Count",
                        "Success ≥50",
                    ],
                    tablefmt="fancy_grid",
                )
            )

            for param_key, dim_subscores in param_scores[rule_id].items():
                print(f"\n    🔧 Param Config: {param_key}")
                table = []
                for dim, vals in dim_subscores.items():
                    stats = self._compute_stats(vals)
                    table.append(
                        [
                            dim,
                            f"{stats['avg']:.2f}",
                            f"{stats['min']:.1f} / {stats['max']:.1f}",
                            f"{stats['std']:.2f}",
                            stats["count"],
                            f"{stats['success_rate_≥50']:.0%}",
                        ]
                    )
                print(
                    tabulate(
                        table,
                        headers=[
                            "Dimension",
                            "Avg",
                            "Min/Max",
                            "Std",
                            "Count",
                            "Success ≥50",
                        ],
                        tablefmt="rounded_outline",
                    )
                )

        return rule_scores  # or return summarized dict if needed

    def pipeline_run_scores(
        self, pipeline_run_id: Optional[int] = None, context: dict = None
    ) -> None:
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
                .filter(EvaluationRuleLinkORM.evaluation_id == score.id)
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
            self.logger.log(
                "PipelineRunScoreSummary",
                {"pipeline_run_id": pipeline_run_id, "total_scores": len(scores)},
            )
