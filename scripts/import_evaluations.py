# scripts/import_evaluations.py
import csv
import json
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.models.evaluation_attribute import EvaluationAttributeORM

# Increase CSV field size limit (safe for Windows)
csv.field_size_limit(1_000_000_000)

# 1. Connect to Postgres
DATABASE_URL = "postgresql+psycopg2://co:co@localhost:5432/co"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# 2. Path to CSV file
CSV_PATH = "C:\\Users\\ernan\\Downloads\\evaluation_export.csv" 

def import_csv():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Wrap reader in tqdm for progress tracking
        for i, row in enumerate(tqdm(reader, desc="Importing evaluations", unit="rows")):
            try:
                # Parse JSON fields
                scores = json.loads(row.get("scores", "[]") or "[]")
                attributes = json.loads(row.get("attributes", "[]") or "[]")

                # Create Evaluation
                evaluation = EvaluationORM(
                    id=int(row["evaluation_id"]),
                    scorable_type=row["scorable_type"],
                    scorable_id=row["scorable_id"],
                    agent_name=row["agent_name"],
                    model_name=row["model_name"],
                    evaluator_name=row["evaluator_name"],
                    strategy=row.get("strategy"),
                    reasoning_strategy=row.get("reasoning_strategy"),
                    embedding_type=row.get("embedding_type"),
                    source=row.get("source"),
                    pipeline_run_id=1,
                    symbolic_rule_id=int(row["symbolic_rule_id"]) if row.get("symbolic_rule_id") else None,
                    extra_data=json.loads(row["extra_data"]) if row.get("extra_data") else None,
                    created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else None,
                )
                session.add(evaluation)
                session.flush()

                # Create Scores
                for s in scores:
                    score = ScoreORM(
                        evaluation_id=evaluation.id,
                        dimension=s["dimension"],
                        score=s.get("score"),
                        weight=s.get("weight"),
                        rationale=s.get("rationale"),
                        source=s.get("source"),
                        prompt_hash=s.get("prompt_hash"),
                    )
                    session.add(score)

                # Create Attributes
                for a in attributes:
                    attr = EvaluationAttributeORM(
                        evaluation_id=evaluation.id,
                        dimension=a["dimension"],
                        source=a["source"],
                        raw_score=a.get("raw_score"),
                        energy=a.get("energy"),
                        q_value=a.get("q"),
                        v_value=a.get("v"),
                        advantage=a.get("advantage"),
                        pi_value=a.get("pi"),
                        entropy=a.get("entropy"),
                        uncertainty=a.get("uncertainty"),
                        td_error=a.get("td_error"),
                        expected_return=a.get("expected_return"),
                        policy_logits=a.get("policy_logits"),
                        extra=a.get("extra"),
                    )
                    session.add(attr)

                # Commit in batches of 500 to avoid RAM blowup
                if i % 500 == 0 and i > 0:
                    session.commit()

            except Exception as e:
                print(f"❌ Error on row {i}: {e}")
                session.rollback()

    # Final commit
    session.commit()
    print("✅ Import complete!")

if __name__ == "__main__":
    import_csv()
