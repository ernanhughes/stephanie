# stephanie/agents/pipeline_comparison.py
import csv
import os
from collections import defaultdict
from datetime import datetime

from sqlalchemy import func

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.analysis.rule_effect_analyzer import RuleEffectAnalyzer
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.models import EvaluationORM, ScoreORM
from stephanie.models.comparison_preference import ComparisonPreferenceORM


class PipelineComparisonAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.print_results = cfg.get("print_results", True)
        self.tags = cfg.get("tags", [])
        self.analyzer = RuleEffectAnalyzer(
            session=self.memory.session, logger=self.logger
        )

    async def run(self, context: dict) -> dict:
        self.logger.log(
            "PipelineComparisonAgentStart",
            {PIPELINE_RUN_ID: context.get(PIPELINE_RUN_ID)},
        )

        if not self.tags:
            raise ValueError(
                "No tags provided for comparison. Please specify tags in the configuration."
            )

        results = self.compare_runs(self.tags, goal_id=context.get("goal_id"))

        if self.print_results:
            print("\n=== Pipeline Comparison Results ===")
            for result in results:
                print(
                    f"Goal ID: {result['goal_id']}, Winner: {result['winner']}, Scores: {result['avg_scores']}"
                )

        self.logger.log("PipelineComparisonAgentEnd", {"output_key": self.output_key})
        return context

    def compare_runs(self, tags: list[str], goal_id: int = None):
        from stephanie.models.comparison_preference import \
            ComparisonPreferenceORM

        """
        Compare multiple sets of pipeline runs by tag across the same goals.
        Store MR.Q-style preferences when there's a clear winner.
        """
        print(f"\n🔍 Comparing Pipelines: {tags}\n")

        # Fetch all runs for each tag
        runs_by_tag = {
            tag: self.memory.pipeline_runs.find({"tag": tag}) for tag in tags
        }

        # Index runs by goal_id
        runs_by_goal = defaultdict(dict)
        for tag, runs in runs_by_tag.items():
            for run in runs:
                if goal_id and run.goal_id != goal_id:
                    continue
                runs_by_goal[run.goal_id][tag] = run

        results = []
        for goal, tag_run_map in runs_by_goal.items():
            if len(tag_run_map) < 2:
                continue  # Skip if fewer than 2 runs to compare

            score_map = {}
            for tag, run in tag_run_map.items():
                score_map[tag] = self.get_multidimensional_score(run.id)

            avg_scores = {
                tag: scores.get("overall", 0.0) for tag, scores in score_map.items()
            }
            best_tag = max(avg_scores, key=avg_scores.get)
            is_tie = list(avg_scores.values()).count(avg_scores[best_tag]) > 1
            winner = "Tie" if is_tie else best_tag

            if winner != "Tie":
                loser = [tag for tag in tags if tag != winner][0]
                preference = ComparisonPreferenceORM(
                    goal_id=goal,
                    preferred_tag=winner,
                    rejected_tag=loser,
                    preferred_run_id=tag_run_map[winner].run_id,
                    rejected_run_id=tag_run_map[loser].run_id,
                    preferred_score=avg_scores[winner],
                    rejected_score=avg_scores[loser],
                    dimension_scores=score_map,
                    reason=f"{winner} outperformed {loser} on score {avg_scores[winner]:.2f} > {avg_scores[loser]:.2f}",
                )
                self.memory.session.add(preference)
                self._save_comparison_preference(
                    goal, winner, loser, tag_run_map, avg_scores, score_map
                )

            results.append(
                {
                    "goal_id": goal,
                    "avg_scores": avg_scores,
                    "winner": winner,
                    "run_ids": {tag: tag_run_map[tag].run_id for tag in tag_run_map},
                    "dimensions": score_map,
                }
            )

        self.memory.session.commit()  # commit all preferences
        self._print_summary(results, tags)

        for r in results:
            if r["winner"] == "Tie":
                continue  # skip ties
            goal_id = r["goal_id"]
            preferred_tag = r["winner"]
            rejected_tag = [t for t in tags if t != preferred_tag][0]

            preferred_run_id = r["run_ids"][preferred_tag]
            rejected_run_id = r["run_ids"][rejected_tag]
            preferred_score = r["avg_scores"][preferred_tag]
            rejected_score = r["avg_scores"][rejected_tag]
            dimension_scores = r["dimensions"]

            self.memory.mrq.insert_or_update_mrq_sample(
                goal_id=goal_id,
                preferred_run_id=preferred_run_id,
                rejected_run_id=rejected_run_id,
                preferred_score=preferred_score,
                rejected_score=rejected_score,
                dimension_scores=dimension_scores,
                source="pipeline_comparison",
            )

        self.export_to_csv(results, tags)
        return results

    def _get_avg_score(self, run_id: str) -> float:
        scores = (
            self.memory.session.query(ScoreORM)
            .filter(ScoreORM.pipeline_run_id == run_id)
            .all()
        )
        if not scores:
            return 0.0
        return sum(s.score for s in scores) / len(scores)

    def _print_summary(self, results, tags):
        from tabulate import tabulate

        headers = ["Goal ID"] + [f"Score {tag}" for tag in tags] + ["Winner"]
        table_data = []

        for r in results:
            row = (
                [r["goal_id"]]
                + [f"{r['avg_scores'].get(tag, 0.0):.2f}" for tag in tags]
                + [r["winner"]]
            )
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

        total = len(results)
        win_counts = {
            tag: sum(1 for r in results if r["winner"] == tag) for tag in tags
        }
        ties = sum(1 for r in results if r["winner"] == "Tie")

        print("\n✅ Summary:")
        for tag in tags:
            print(f"  {tag} wins: {win_counts[tag]}")
        print(f"  Ties: {ties} (Total comparisons: {total})")

    def export_to_csv(self, results: list[dict], tags: list[str], filename: str = None):
        """
        Exports the comparison results to a CSV file.
        """
        filename = (
            filename
            or f"pipeline_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        filepath = os.path.join("reports", filename)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = ["goal_id"] + [f"{tag}_score" for tag in tags] + ["winner"]
            writer.writerow(headers)

            for r in results:
                row = (
                    [r["goal_id"]]
                    + [r["avg_scores"].get(tag, 0.0) for tag in tags]
                    + [r["winner"]]
                )
                writer.writerow(row)

        print(f"\n📄 Exported comparison to {filepath}")

    def get_multidimensional_score(self, pipeline_run_id: int) -> dict:
        """
        Fetch and average all score values for a given pipeline run via its evaluations.
        Updated to support normalized target_type/target_id.
        """

        # Step 1: Get latest evaluation per unique target
        subquery = (
            self.memory.session.query(
                EvaluationORM.target_type,
                EvaluationORM.target_id,
                func.max(EvaluationORM.id).label("latest_eval_id"),
            )
            .filter(EvaluationORM.pipeline_run_id == pipeline_run_id)
            .group_by(EvaluationORM.target_type, EvaluationORM.target_id)
            .subquery()
        )

        # Step 2: Get those evaluation IDs
        latest_eval_ids = [
            row.latest_eval_id for row in self.memory.session.query(subquery).all()
        ]
        if not latest_eval_ids:
            return {}

        # Step 3: Fetch scores and average by dimension
        dim_totals = defaultdict(list)
        for evaluation_id in latest_eval_ids:
            scores = (
                self.memory.session.query(ScoreORM)
                .filter(ScoreORM.evaluation_id == evaluation_id)
                .all()
            )
            for s in scores:
                dim_totals[s.dimension].append(s.score)

        averaged = {dim: sum(vals) / len(vals) for dim, vals in dim_totals.items()}
        if averaged:
            averaged["overall"] = sum(averaged.values()) / len(averaged)

        return averaged

    def _save_comparison_preference(
        self, goal_id, preferred_tag, rejected_tag, tag_run_map, avg_scores, score_map
    ):
        preferred_run_id = tag_run_map[preferred_tag].run_id
        rejected_run_id = tag_run_map[rejected_tag].run_id

        existing = (
            self.memory.session.query(ComparisonPreferenceORM)
            .filter_by(
                goal_id=goal_id,
                preferred_run_id=preferred_run_id,
                rejected_run_id=rejected_run_id,
                source="pipeline_comparison",
            )
            .first()
        )

        data = {
            "goal_id": goal_id,
            "preferred_tag": preferred_tag,
            "rejected_tag": rejected_tag,
            "preferred_run_id": preferred_run_id,
            "rejected_run_id": rejected_run_id,
            "preferred_score": avg_scores[preferred_tag],
            "rejected_score": avg_scores[rejected_tag],
            "dimension_scores": score_map,
            "reason": f"{preferred_tag} outperformed {rejected_tag} ({avg_scores[preferred_tag]:.2f} > {avg_scores[rejected_tag]:.2f})",
            "source": "pipeline_comparison",
        }

        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            preference = ComparisonPreferenceORM(**data)
            self.memory.session.add(preference)
