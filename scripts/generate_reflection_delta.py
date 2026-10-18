import pandas as pd
from sqlalchemy import text
from stephanie.orm.base import create_database_engine_from_hydra

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

def main():
    engine = create_database_engine_from_hydra("config")
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
    finally:
        engine.dispose()

    df.to_json("reflection_deltas.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
