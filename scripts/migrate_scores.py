import json
from sqlalchemy.orm import Session
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def migrate_scores_from_json(session: Session, logger=None):
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

    session.commit()
    if logger:
        logger.log("ScoreMigration", {"inserted": count_inserted})
    print(f"✅ Migration complete. Inserted {count_inserted} dimension scores.")


engine = create_engine("postgresql://co:co@localhost/co")
Session = sessionmaker(bind=engine)
session = Session()

migrate_scores_from_json(session)