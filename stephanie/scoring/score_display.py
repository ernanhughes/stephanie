# stephanie/scoring/score_display.py
from tabulate import tabulate


class ScoreDisplay:
    @staticmethod
    def show(results, weighted_score):
        table_data = [
            [
                dim_name,
                f"{dim_data['score']:.2f}",
                dim_data.get("weight", 1.0),
                dim_data.get("rationale", "")[:60]
            ]
            for dim_name, dim_data in results.items()
        ]
        table_data.append(["FINAL", f"{weighted_score:.2f}", "-", "Weighted average"])

        print("\n📊 Dimension Scores Summary")
        print(tabulate(
            table_data,
            headers=["Dimension", "Score", "Weight", "Rationale (preview)"],
            tablefmt="fancy_grid"
        ))
