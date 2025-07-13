# stephanie/tools/view_ranking_trace.py
# tools/view_ranking_trace.py
from collections import defaultdict

import matplotlib.pyplot as plt
import psycopg2
from tabulate import tabulate

DB_CONFIG = dict(dbname="co", user="co", password="co", host="localhost")


def fetch_ranking_trace(run_id=None):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            if run_id:
                cur.execute(
                    """
                    SELECT winner, loser, explanation, created_at
                    FROM ranking_trace
                    WHERE run_id = %s
                    ORDER BY created_at
                """,
                    (run_id,),
                )
            else:
                cur.execute("""
                    SELECT winner, loser, explanation, created_at
                    FROM ranking_trace
                    ORDER BY created_at DESC LIMIT 50
                """)
            return cur.fetchall()


def fetch_elo_scores(run_id=None):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            if run_id:
                cur.execute(
                    """
                    SELECT hypothesis, score, created_at
                    FROM elo_ranking_log
                    WHERE run_id = %s
                    ORDER BY created_at
                """,
                    (run_id,),
                )
            else:
                cur.execute("""
                    SELECT hypothesis, score, created_at
                    FROM elo_ranking_log
                    ORDER BY created_at DESC LIMIT 50
                """)
            return cur.fetchall()


def display_top_ranked(run_id=None):
    scores = fetch_elo_scores(run_id)
    latest_scores = {}
    for hypo, score, ts in scores:
        latest_scores[hypo] = score
    sorted_scores = sorted(latest_scores.items(), key=lambda x: x[1], reverse=True)
    print("\nTop-Ranked Hypotheses:\n")
    print(tabulate(sorted_scores, headers=["hypotheses", "ELO Score"], tablefmt="grid"))


def plot_elo_evolution(run_id=None):
    scores = fetch_elo_scores(run_id)
    time_series = defaultdict(list)
    for hypo, score, ts in scores:
        time_series[hypo].append((ts, score))

    plt.figure(figsize=(10, 6))
    for hypo, points in time_series.items():
        times, scores = zip(*points)
        plt.plot(times, scores, label=hypo[:40] + ("..." if len(hypo) > 40 else ""))

    plt.xlabel("Time")
    plt.ylabel("ELO Score")
    plt.title("ELO Evolution Over Time")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def main():
    run_id = input("Enter run_id (or leave blank for latest): ").strip() or None
    print("\n--- Ranking Trace ---")
    rows = fetch_ranking_trace(run_id)
    print(
        tabulate(
            rows, headers=["Winner", "Loser", "Explanation", "Time"], tablefmt="grid"
        )
    )

    print("\n--- Top-Ranked Hypotheses ---")
    display_top_ranked(run_id)

    print("\n--- Plotting ELO Evolution ---")
    plot_elo_evolution(run_id)


if __name__ == "__main__":
    main()
