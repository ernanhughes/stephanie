import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

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
        with psycopg2.connect(**DB_CONFIG) as conn:
            df = pd.read_sql_query(AGGREGATE_QUERY, conn)

        print("\n🧠 Reasoning Pattern Summary:")
        print(df)

        df.to_csv("pattern_summary.csv", index=False)
        print("\n✅ Summary saved to pattern_summary.csv")

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    summarize_pattern_stats()
