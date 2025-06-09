from typing import List, Optional

from sqlalchemy.orm import Session

from co_ai.models.score import ScoreORM


class ScoreStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "scores"
        self.table_name = "scores"

    def add_score(self, score: ScoreORM) -> ScoreORM:
        self.session.add(score)
        self.session.commit()
        self.session.refresh(score)
        return score

    def add_scores_bulk(self, scores: List[ScoreORM]):
        self.session.add_all(scores)
        self.session.commit()

    def get_scores_for_evaluation(self, evaluation_id: int) -> List[ScoreORM]:
        return (
            self.session.query(ScoreORM)
            .filter_by(evaluation_id=evaluation_id)
            .order_by(ScoreORM.dimension.asc())
            .all()
        )

    def get_scores_for_hypothesis(self, hypothesis_id: int) -> List[ScoreORM]:
        return (
            self.session.query(ScoreORM)
            .filter(ScoreORM.evaluation.has(hypothesis_id=hypothesis_id))
            .order_by(ScoreORM.dimension.asc())
            .all()
        )

    def get_scores_by_dimension(self, dimension: str, top_k: int = 100) -> List[ScoreORM]:
        return (
            self.session.query(ScoreORM)
            .filter_by(dimension=dimension)
            .order_by(ScoreORM.score.desc().nullslast())
            .limit(top_k)
            .all()
        )

    def delete_scores_for_evaluation(self, evaluation_id: int):
        self.session.query(ScoreORM).filter_by(evaluation_id=evaluation_id).delete()
        self.session.commit()

    def get_all(self, limit: Optional[int] = None) -> List[ScoreORM]:
        query = self.session.query(ScoreORM).order_by(ScoreORM.id.desc())
        if limit:
            query = query.limit(limit)
        return query.all()

    def get_by_id(self, score_id: int) -> Optional[ScoreORM]:
        return self.session.query(ScoreORM).filter_by(id=score_id).first()

    def get_by_evaluation_ids(self, evaluation_ids: list[int]) -> list[ScoreORM]:
        if not evaluation_ids:
            return []
        try:
            return (
                self.session.query(ScoreORM)
                .filter(ScoreORM.evaluation_id.in_(evaluation_ids))
                .all()
            )
        except Exception as e:
            if self.logger:
                self.logger.log("ScoreStoreError", {
                    "method": "get_by_evaluation_ids",
                    "error": str(e),
                    "evaluation_ids": evaluation_ids,
                })
            return []