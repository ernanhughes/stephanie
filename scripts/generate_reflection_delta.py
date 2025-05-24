import psycopg2
import pandas as pd
import os

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

query = """
SELECT 
    id, 
    goal_id, 
    run_id_a, 
    run_id_b, 
    score_a, 
    score_b, 
    score_delta, 
    pipeline_diff, 
    strategy_diff, 
    model_diff, 
    rationale_diff, 
    created_at
FROM reflection_deltas
WHERE score_a IS NOT NULL AND score_b IS NOT NULL
  AND ABS(score_b - score_a) >= 0.05
ORDER BY created_at DESC;
"""

with psycopg2.connect(**DB_CONFIG) as conn:
    df = pd.read_sql_query(query, conn)

df = pd.read_sql_query(query, conn)
df.to_json("reflection_deltas.jsonl", orient="records", lines=True)
conn.close()
