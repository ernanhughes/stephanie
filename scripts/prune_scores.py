from sqlalchemy.orm import Session
from stephanie.models import ScoreORM, EvaluationORM

def prune_zero_scores_and_orphan_evaluations(session: Session, document_id: int, logger=None):
    # Step 1: Find and delete zero scores
    zero_scores = (
        session.query(ScoreORM)
        .filter(ScoreORM.document_id == document_id)
        .filter(ScoreORM.score == 0)
        .all()
    )

    zero_score_count = len(zero_scores)
    deleted_eval_ids = set()

    for score in zero_scores:
        deleted_eval_ids.add(score.evaluation_id)
        session.delete(score)

    session.commit()

    if logger:
        logger.log("ZeroScoresPruned", {
            "document_id": document_id,
            "score_count": zero_score_count
        })

    # Step 2: Delete evaluations with no scores left
    for eval_id in deleted_eval_ids:
        score_exists = (
            session.query(ScoreORM)
            .filter(ScoreORM.evaluation_id == eval_id)
            .first()
        )
        if not score_exists:
            eval_entry = session.query(EvaluationORM).filter_by(id=eval_id).first()
            if eval_entry:
                session.delete(eval_entry)
                if logger:
                    logger.log("EvaluationPruned", {
                        "evaluation_id": eval_id,
                        "document_id": document_id
                    })

    session.commit()
