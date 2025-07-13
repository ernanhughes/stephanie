# stephanie/agents/pipeline_judge.py
import csv
import os
import re
from datetime import datetime

from tabulate import tabulate

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.analysis.rule_analytics import RuleAnalytics
from stephanie.analysis.rule_effect_analyzer import RuleEffectAnalyzer
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType


class PipelineJudgeAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.print_results = cfg.get("print_results", True)

    async def run(self, context: dict) -> dict:
        self.logger.log(
            "PipelineJudgeAgentStart", {PIPELINE_RUN_ID: context.get(PIPELINE_RUN_ID)}
        )
        hypotheses = context.get("scored_hypotheses", []) or context.get(
            "hypotheses", []
        )
        documents = context.get("documents", [])

        self.logger.log(
            "DocumentsToJudge",
            {
                "Hypotheses": {
                    "count": len(hypotheses),
                    "source": "scored_hypotheses"
                    if context.get("scored_hypotheses")
                    else "hypotheses",
                },
                "Documents": {"count": len(hypotheses), "source": "documents"},
            },
        )

        for doc in documents:
            scorable = Scorable(
                id=doc.get("id"),
                text=doc.get("summary", ""),
                target_type=TargetType.DOCUMENT,
            )
            score_result = self.score_item(
                scorable=scorable,
                context=context,
                metrics="pipeline_judge",
            )
            self.logger.log(
                "HypothesisJudged",
                {"hypothesis_id": doc.get("id"), "score": score_result.to_dict()},
            )

        for hypo in hypotheses:
            scorable = Scorable(
                id=hypo.get("id"),
                text=hypo.get("text", ""),
                target_type=TargetType.HYPOTHESIS,
            )
            score_result = self.score_item(
                scorable=scorable,
                context=context,
                metrics="pipeline_judge",
            )
            self.logger.log(
                "HypothesisJudged",
                {"hypothesis_id": hypo.get("id"), "score": score_result.to_dict()},
            )

        self.report_rule_analytics(context)
        self.run_rule_effects_evaluation(context)

        self.logger.log("PipelineJudgeAgentEnd", {"output_key": self.output_key})
        return context

    def report_rule_analytics(self, context: dict):
        analytics = RuleAnalytics(db=self.memory, logger=self.logger)
        results = analytics.analyze_rules_for_run(context.get(PIPELINE_RUN_ID))

        if results and isinstance(results, list) and self.print_results:
            print("\n=== Rule Analytics Summary ===")
            print(f"{'Rule ID':<10}{'Applications':<15}{'Avg Score':<12}")
            print("-" * 40)
            for result in results:
                rule_id = result.get("rule_id")
                count = result.get("count", 0)
                avg_score = result.get("avg_score", 0.0)
                print(f"{rule_id:<10}{count:<15}{avg_score:<12.2f}")
            print("-" * 40)

    def run_rule_effects_evaluation(self, context: dict, output_dir: str = "./reports"):
        analyzer = RuleEffectAnalyzer(session=self.memory.session, logger=self.logger)
        summary = analyzer.analyze(context.get(PIPELINE_RUN_ID))

        # Sort by average score descending
        top_rules = sorted(
            summary.items(), key=lambda x: x[1]["avg_score"], reverse=True
        )

        # Prepare CSV rows
        rows = []
        for rule_id, data in top_rules:
            rows.append(
                [
                    rule_id,
                    f"{data['avg_score']:.2f}",
                    data["count"],
                    data["min"],
                    data["max"],
                    f"{data['std']:.2f}",
                    f"{data['success_rate']:.2%}",
                ]
            )

        # Define headers
        headers = [
            "Rule ID",
            "Avg Score",
            "Count",
            "Min",
            "Max",
            "Std Dev",
            "Success Rate ≥50%",
        ]

        # Ensure export directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Compose output filename with timestamp
        run_id = context.get(PIPELINE_RUN_ID, "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rule_effects_{run_id}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)

        # Write to CSV
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)

        print(f"\n✅ Rule effect summary saved to: {filepath}")

        # Optional: also return summary if needed
        return summary

    def run_rule_effects_evaluation_console(self, context: dict):
        from tabulate import tabulate

        analyzer = RuleEffectAnalyzer(session=self.memory.session, logger=self.logger)
        summary = analyzer.analyze(context.get(PIPELINE_RUN_ID))

        # Sort by average score descending
        top_rules = sorted(
            summary.items(), key=lambda x: x[1]["avg_score"], reverse=True
        )

        # Prepare table data
        table_data = []
        for rule_id, data in top_rules[:5]:
            table_data.append(
                [
                    rule_id,
                    f"{data['avg_score']:.2f}",
                    data["count"],
                    f"{data['min']} / {data['max']}",
                    f"{data['std']:.2f}",
                    f"{data['success_rate']:.2%}",
                ]
            )

        # Define headers
        headers = [
            "Rule ID",
            "Avg Score",
            "Count",
            "Min / Max",
            "Std Dev",
            "Success Rate ≥50",
        ]

        # Print table
        print("\n📈 Top Performing Rules:")
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

        analyzer.pipeline_run_scores(context=context)
