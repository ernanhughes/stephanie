import argparse
import json
from sqlalchemy.orm import Session
from stephanie.orm.evaluation import EvaluationORM
from stephanie.orm.score import ScoreORM
from sqlalchemy.orm import sessionmaker
from stephanie.orm.base import create_database_engine_from_hydra


def migrate_scores_from_json(session: Session, *, apply: bool = False):
    evaluations = session.query(EvaluationORM).all()
    count_inserted = 0

    for eval_row in evaluations:
        evaluation_id = eval_row.id
        extra_data = eval_row.scores

        if not extra_data or not isinstance(extra_data, dict):
            try:
                extra_data = json.loads(extra_data)
            except Exception:
                continue

        dimensions = extra_data.get("dimensions", {})
        for dim_name, dim_data in dimensions.items():
            score = dim_data.get("score")
            weight = dim_data.get("weight", 1.0)
            rationale = dim_data.get("rationale", "")

            if score is None:
                continue

            score_row = ScoreORM(
                evaluation_id=evaluation_id,
                dimension=dim_name,
                score=score,
                weight=weight,
                rationale=rationale
            )
            session.add(score_row)
            count_inserted += 1

    if apply:
        session.commit()
        print(f"Applied migration; inserted {count_inserted} dimension scores.")
    else:
        session.rollback()
        print(f"Dry run: {count_inserted} dimension scores would be inserted; pass --apply to write.")
    return count_inserted


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy JSON scores into score rows")
    parser.add_argument("--apply", action="store_true", help="Commit inserts (default: dry run)")
    args = parser.parse_args()

    engine = create_database_engine_from_hydra("config")
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        migrate_scores_from_json(session, apply=args.apply)
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    main()
