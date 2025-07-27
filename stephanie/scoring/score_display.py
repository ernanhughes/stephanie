# stephanie/scoring/score_display.py
from tabulate import tabulate


class ScoreDisplay:
    @staticmethod
    def show(scorable, results, weighted_score):
        table_data = [
            [
                dim_name,
                f"{dim_data['score']:.2f}",
                dim_data.get("weight", 1.0),
                dim_data.get("rationale", "")[:60],
            ]
            for dim_name, dim_data in results.items()
        ]
        source = "Unknown"
        try:
            table_data.append(["FINAL", f"{weighted_score:.2f}", "-", "Weighted average"])
            _, value = next(iter(results.items()))
            source = value.get("source", "Unknown")
        except StopIteration:
            pass
        print(f"\n📊 {source} Dimension Scores {scorable.target_type}:{scorable.id} Summary")
        print(
            tabulate(
                table_data,
                headers=["Dimension", "Score", "Weight", "Rationale (preview)"],
                tablefmt="fancy_grid",
            )
        )
