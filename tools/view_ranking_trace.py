# tools/view_ranking_trace.py
import psycopg2
from tabulate import tabulate

DB_CONFIG = dict(
    dbname="co",
    user="co",
    password="co",
    host="localhost"
)

def fetch_ranking_trace(run_id=None):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            if run_id:
                cur.execute("""
                    SELECT winner, loser, explanation, created_at
                    FROM ranking_trace
                    WHERE run_id = %s
                    ORDER BY created_at
                """, (run_id,))
            else:
                cur.execute("""
                    SELECT winner, loser, explanation, created_at
                    FROM ranking_trace
                    ORDER BY created_at DESC LIMIT 50
                """)
            return cur.fetchall()

def main():
    run_id = input("Enter run_id (or leave blank for latest): ").strip() or None
    rows = fetch_ranking_trace(run_id)
    print("\nRanking Trace Results:\n")
    print(tabulate(rows, headers=["Winner", "Loser", "Explanation", "Time"], tablefmt="grid"))

if __name__ == "__main__":
    main()
