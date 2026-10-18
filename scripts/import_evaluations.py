# scripts/import_evaluations.py
import csv
import json
import argparse
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from stephanie.orm.evaluation import EvaluationORM
from stephanie.orm.score import ScoreORM
from stephanie.orm.evaluation_attribute import EvaluationAttributeORM
from stephanie.orm.base import create_database_engine_from_hydra

# Increase CSV field size limit (safe for Windows)
csv.field_size_limit(1_000_000_000)

def import_csv(session, csv_path: str, *, apply: bool = False):
    imported = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
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
                if apply:
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
                imported += 1

                # Commit in batches of 500 to avoid RAM blowup
                if apply and i % 500 == 0 and i > 0:
                    session.commit()

            except Exception as e:
                print(f"❌ Error on row {i}: {e}")
                session.rollback()

    if apply:
        session.commit()
        print(f"Import complete; applied {imported} evaluations.")
    else:
        session.rollback()
        print(f"Dry run: {imported} evaluations validated; pass --apply to write.")
    return imported


def main():
    parser = argparse.ArgumentParser(description="Import evaluation CSV rows")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--apply", action="store_true", help="Commit imports (default: dry run)")
    args = parser.parse_args()

    engine = create_database_engine_from_hydra("config")
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        import_csv(session, args.csv_path, apply=args.apply)
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    main()
