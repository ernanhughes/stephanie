import pandas as pd
from sqlalchemy import text
from stephanie.orm.base import create_database_engine_from_hydra

AGGREGATE_QUERY = """
SELECT 
    model_name,
    dimension,
    label,
    COUNT(*) AS count
FROM cot_pattern_stats
GROUP BY model_name, dimension, label
ORDER BY model_name, dimension, count DESC;
"""

def summarize_pattern_stats():
    try:
        engine = create_database_engine_from_hydra("config")
        try:
            with engine.connect() as conn:
                df = pd.read_sql_query(text(AGGREGATE_QUERY), conn)
        finally:
            engine.dispose()

        print("\n🧠 Reasoning Pattern Summary:")
        print(df)

        df.to_csv("pattern_summary.csv", index=False)
        print("\n✅ Summary saved to pattern_summary.csv")

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    summarize_pattern_stats()
